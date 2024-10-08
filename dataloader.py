import torch
import numpy as np
import torch.utils.data as data
import pandas as pd
import json
from collections import defaultdict

# Constants
UCI_ADULT_PATH = "data/datasets/uci_adult/"
COMPAS_PATH = "data/datasets/compas/"
LAW_SCHOOL_PATH = "data/datasets/law_school/"
CRIME_PATH = "data/datasets/crime/"

pure_reg=['crime', 'synthetic']


class loadDataset(data.Dataset):
    def __init__(self, dataset, train_or_test, embedding_size=None, without_sensitive_attributes=True):
        self.wosa = without_sensitive_attributes
        if dataset in pure_reg:
            self.categorical_data = None
            self.categorical_embedding_sizes = 0
            root_path = './data/datasets/'
            if dataset == "crime":
                df = pd.read_csv(f'{root_path}crime/{train_or_test}.csv')
                df = df.fillna(0)
                # process race
                B = "racepctblack"
                W = "racePctWhite"
                A = "racePctAsian"
                H = "racePctHisp"
                sens_features = [2, 3, 4, 5]
                df_sens = df.iloc[:, sens_features]
                maj = df_sens.apply(pd.Series.idxmax, axis=1)
                # remap the values of maj
                a = maj.map({B: 0, W: 1, A: 0, H: 0})
                df['race'] = a
                df = df.drop(H, axis=1)
                df = df.drop(B, axis=1)
                df = df.drop(W, axis=1)
                df = df.drop(A, axis=1)

                self.numerical_data = df
                self.sensitive_attributes = ['race', 'medIncome', 'householdsize', 'medFamInc']
                self.normalize(regression=True)

                # creating labels using crime rate
                Y = self.numerical_data['ViolentCrimesPerPop']
                self.numerical_data = self.numerical_data.drop('ViolentCrimesPerPop', axis=1)
                self.target_data = torch.tensor(Y).reshape([-1, 1])

                sensitive_threshold = 0.5
                # binarize sensitive attributes
                for sa in self.sensitive_attributes:
                    self.numerical_data.loc[self.numerical_data[sa] > sensitive_threshold, sa] = 1
                    self.numerical_data.loc[self.numerical_data[sa] <= sensitive_threshold, sa] = 0
                # self.sensitive_attributes = np.array(df[sensitive_attributes]).reshape([-1, len(sensitive_attributes)])
                # subgroup
                self.set_subgroups(regression=True)
                # self.numerical_data["y"] = np.array(self.target_data)
                # np.save("crime.npy", self.numerical_data)
                self.numerical_data = torch.tensor(self.numerical_data.values)
            elif dataset == "synthetic":
                synthetic_dataset = np.load(root_path + dataset + '/' + train_or_test + ".npy")
                self.target_data = torch.tensor(synthetic_dataset[:, 6]).reshape([-1, 1])
                self.idx2group = torch.tensor(synthetic_dataset[:, 5])
                self.subgroup_indexes = [[], []]
                for i in range(len(synthetic_dataset)):
                    if self.idx2group[i] > 0:
                        self.subgroup_indexes[0].append(i)
                    else:
                        self.subgroup_indexes[1].append(i)
                self.numerical_data = torch.tensor(synthetic_dataset[:, :5])
            else:
                raise AttributeError("Unknown dataset")
            self.numerical_sizes = self.numerical_data.shape[1]
        else:
            self.numerical_data, self.dataset_stats = get_dataset_stats(
                dataset, train_or_test
            )

            self.mean_std = self.dataset_stats["mean_std"]
            self.numerical_sizes = len(self.mean_std.keys())
            self.vocabulary = self.dataset_stats["vocabulary"]
            self.target_column_name = self.dataset_stats["target_column_name"]
            self.target_column_positive_value = self.dataset_stats[
                "target_column_positive_value"
            ]
            self.sensitive_column_names = self.dataset_stats["sensitive_column_names"]
            self.sensitive_column_values = self.dataset_stats["sensitive_column_values"]
            self.embedding_size = embedding_size

            self.prepare_dataframe()
            self.binarize()
            self.normalize(regression=False)

            if embedding_size:
                self.calculate_embedding()
            self.set_subgroups(regression=False)
            self.stack_data()

    def prepare_dataframe(self):
        """
        Ensures all columns have the correct dtype.
        """
        # Rename dataframe columns and replace empty string with 'unk'
        self.numerical_data.columns = self.dataset_stats["feature_names"]
        self.numerical_data.fillna("unk", inplace=True)

        # Change dtype to categorical for all category columns
        for category in self.vocabulary.keys():
            self.numerical_data[category] = self.numerical_data[category].astype("category")

    def binarize(self):
        """
        Ensures target data and protected features are binary.
        """
        # Binarize target variables.
        self.numerical_data[self.target_column_name] = self.numerical_data[self.target_column_name].astype("category")
        self.numerical_data[self.target_column_name] = (self.numerical_data[
                                                       self.target_column_name] == self.target_column_positive_value) * 1
        self.target_data = torch.Tensor(self.numerical_data[self.target_column_name].values)

        # Binarize protected features. 
        for sensitive_column_name, sensitive_column_value in zip(
                self.sensitive_column_names, self.sensitive_column_values
        ):
            self.numerical_data[sensitive_column_name] = (
                                                            self.numerical_data[
                                                                sensitive_column_name] == sensitive_column_value
                                                    ) * 1
        self.protected_data = torch.Tensor(
            self.numerical_data[self.sensitive_column_names].values
        )

    def normalize(self, regression=True):
        """
        Ensures numerical data has zero mean and variance.
        """
        if regression:
            for column in self.numerical_data:
                if column != 'race':
                    mean = np.mean(self.numerical_data[column])
                    std = np.std(self.numerical_data[column])
                    self.numerical_data[column] = (self.numerical_data[column] - mean) / std
        else:
            for key, value in self.mean_std.items():
                mean = value[0]
                std = value[1]
                self.numerical_data[key] = (self.numerical_data[key] - mean) / std

    def calculate_embedding(self):
        """
        Calculates the embedding size for categorical data.
        """
        self.categorical_embedding_sizes = [
            (len(vocab) + 1, self.embedding_size)
            for cat, vocab in self.vocabulary.items()
            if cat not in self.sensitive_column_names and cat != self.target_column_name
        ]

    def set_subgroups(self, regression=True):
        """
        Use the cartesian product to get subgroups of protected groups.
        for example the subgroups: [black male].
        """
        if regression:
            self.protected_data = self.numerical_data[self.sensitive_attributes]
            sensitive_group_len = len(self.sensitive_attributes)
            self.idx2group = []
            self.subgroup_indexes = [[] for _ in range(2 ** sensitive_group_len)]
            for i in range(len(self.protected_data)):
                group_id = int(sum([self.protected_data.loc[i][idx] * (2 ** idx) for idx in range(sensitive_group_len)]))
                self.idx2group.append(group_id)
                self.subgroup_indexes[group_id].append(i)
            self.subgroup_indexes = [sub for sub in self.subgroup_indexes if len(sub) > 0]
            self.protected_data = torch.tensor(np.array(self.protected_data))
            subgroup_counts = [len(c) for c in self.subgroup_indexes]
            self.subgroup_minority = np.argmin(np.array(subgroup_counts))
            for sa in self.sensitive_attributes:
                self.numerical_data = self.numerical_data.drop(sa, axis=1)
        else:
            opt = self.protected_data.unique().numpy()
            combinations = np.transpose([np.tile(opt, len(opt)),
                                         np.repeat(opt, len(opt))])
            subgroups = [np.where((self.numerical_data
                                   [self.sensitive_column_names[0]] == comb[0])
                                  & (self.numerical_data[self.sensitive_column_names[1]] == comb[1]), 1, 0)
                         for idx, comb in enumerate(combinations)]

            self.subgroups = pd.DataFrame(subgroups).transpose()

            # Get the minority subgroup (subgroup that is least supported).
            # [male non-black, female non-black, male black, female black]
            # [00, 10, 01, 11]
            subgroup_counts = [np.sum(self.subgroups[c]) for c in range(self.subgroups.shape[1])]
            self.subgroup_minority = np.argmin(np.array(subgroup_counts))

            # Get the indexes of the dataframe rows that correspond to each subgroup.
            subgroup_indexes = []
            for col in range(len(self.subgroups.columns)):
                subgroup_indexes.append(self.subgroups.index
                                        [self.subgroups[col] == 1].tolist())
            self.subgroup_indexes = subgroup_indexes
            self.idx2group = self.subgroups.idxmax(axis=1).values
        print(f"current dataset minor ratio is {subgroup_counts[self.subgroup_minority] / sum(subgroup_counts)}")


    def stack_data(self):
        """
        Change categorical data to one-hot encoded tensors.
        """
        one_hot_encoded = [
            self.numerical_data[feature].cat.codes.values
            for feature in self.vocabulary.keys()
            if feature not in self.sensitive_column_names and feature != self.target_column_name
        ]
        self.categorical_data = torch.tensor(
            np.stack(one_hot_encoded, 1), dtype=torch.int64
        )

        # Stack numerical data int o tensors.
        numerical_data = np.stack(
            [self.numerical_data[col].values for col in self.mean_std.keys()], 1
        )
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float)

    def __getitem__(self, idx):
        """
        Returns one data instance from dataframe.
        """
        numerical_data = self.numerical_data[idx]
        target_data = self.target_data[idx].reshape(-1).float()

        if self.categorical_data is None:
            return None, numerical_data, target_data
        categorical_data = self.categorical_data[idx]

        return categorical_data, numerical_data, target_data

    def __len__(self):
        return len(self.numerical_data)

    @property
    def vocab_size(self):
        return self._vocab_size


def get_dataset_stats(dataset, train_or_test):
    """
    Returns input feature values for each dataset.

    args:
        dataset: the dataset to load (compas, law_school or uci_adult)
        train_or_test: string, specifies either train or test set
    """
    if dataset == "compas":
        data_path = COMPAS_PATH
    elif dataset == "law_school":
        data_path = LAW_SCHOOL_PATH
    elif dataset == "uci_adult":
        data_path = UCI_ADULT_PATH
    else:
        raise AttributeError("Unknown dataset")

    # Read the dataframe.
    dataframe = pd.read_csv(
        data_path + train_or_test + ".csv", header=None
    )

    # Read json file to determine which columns of data to use.
    with open(data_path + "dataset_stats.json") as f:
        dataset_stats = json.load(f)

    # Read json files to distinguish between categorical/numerical data
    with open(data_path + "vocabulary.json") as f:
        dataset_stats["vocabulary"] = json.load(f)
    with open(data_path + "mean_std.json") as f:
        dataset_stats["mean_std"] = json.load(f)

    return dataframe, dataset_stats


class TensorBoardLogger(object):
    def __init__(self, summary_writer, avg_window=5, name=None):
        """
        Class that summarizes some logging code for TensorBoard.
        Open with "tensorboard --logdir logs/" in terminal.
        
        args:
            summary_writer: Summary Writer object from torch.utils.tensorboard.
            avg_window: How often to update the logger. 
            name: Tab name in TensorBoard's scalars.
        """
        self.summary_writer = summary_writer
        if name is None:
            self.name = ""
        else:
            self.name = name + "/"

        self.value_dict = defaultdict(lambda: 0)
        self.steps = defaultdict(lambda: 0)
        self.global_step = 0
        self.avg_window = avg_window

    def add_values(self, log_dict):
        """
        Function for adding a dictionary of logging values to this logger.

        args:
            log_dict:Dictionary of string to Tensor with the values to plot.
        """
        self.global_step += 1

        for key, val in log_dict.items():
            # Detatch if necissary
            if torch.is_tensor(val):
                val = val.detach().cpu()
            self.value_dict[key] += val
            self.steps[key] += 1

            # Plot to TensorBoard every avg_window steps
            if self.steps[key] >= self.avg_window:
                avg_val = self.value_dict[key] / self.steps[key]
                self.summary_writer.add_scalar(
                    self.name + key, avg_val, global_step=self.global_step
                )
                self.value_dict[key] = 0
                self.steps[key] = 0
