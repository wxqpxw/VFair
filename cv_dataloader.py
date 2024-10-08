import os.path

import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

CELEBA_PATH = "data/datasets/celebA/"
AGEDB_PATH = "data/datasets/AgeDB/"
sensitive_att = ["Male", "Young"]
target_att = {
    "beard": ["5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard"],
    "hair_color": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
}


class loadCVDataset(data.Dataset):
    def __init__(self, dataset, train_or_test="train", target="hair_color"):
        self.dataset_name = dataset
        if dataset == "celebA":

            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((198, 168)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((198, 168)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.data_path = CELEBA_PATH
            self.phase = train_or_test
            target_list = target_att[target]
            cols = sensitive_att + target_list + ["filenames", "partition"]
            dataframe = pd.read_csv(self.data_path + "all.csv", usecols=cols)

            if train_or_test == "train":
                self.data = dataframe.loc[dataframe["partition"] == 0]
            else:
                self.data = dataframe.loc[dataframe["partition"] == 2]

            for att in sensitive_att + target_list:
                self.data.loc[:, att] = (self.data.loc[:, att] + 1) // 2

            # del data with no label and multi-label
            self.data.loc[:, "Count"] = self.data.loc[:, target_list].sum(axis=1)
            self.data = self.data.loc[self.data["Count"] == 1]

            print(f"after filter, total length is {len(self.data)}")

            # subgroup
            grouped = self.data.groupby(sensitive_att).indices
            self.subgroup_indexes = []

            # 16
            for name, idx in grouped.items():
                self.subgroup_indexes.append(idx)
                # print(str(name) + str(len(idx)))
                # print(self.data.iloc[idx, :].loc[:, target_list].sum())
            subgroup_cnt = [len(g) for g in self.subgroup_indexes]
            self.subgroup_minority = np.argmin(np.array(subgroup_cnt))
            # idx2group
            idx = [0, 0, 0, 0]
            self.idx2group = []
            for i in range(len(self.data)):
                for j in range(4):
                    if idx[j] < subgroup_cnt[j] and self.subgroup_indexes[j][idx[j]] == i:
                        self.idx2group.append(j)
                        idx[j] += 1
                        break
            print(f"current dataset minor ratio is {subgroup_cnt[self.subgroup_minority] / sum(subgroup_cnt)}")

            # target
            self.target_hot = torch.tensor(self.data.loc[:, target_list].values)
            self.target_num = torch.argmax(self.target_hot, dim=1)
        elif dataset == "ageDB":

            self.train_transform = transforms.Compose([
                transforms.Resize((140, 140)),
                transforms.ToTensor(),
                transforms.RandomCrop((128, 128)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.test_transform = transforms.Compose([
                transforms.Resize((140, 140)),
                transforms.ToTensor(),
                transforms.CenterCrop((128, 128)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.data_path = AGEDB_PATH
            self.phase = train_or_test
            self.subgroup_indexes = [[], []]
            self.idx2group = []
            dataframe = pd.read_csv(os.path.join(self.data_path, train_or_test + '.csv'))

            self.data = dataframe[['filenames']]
            self.target_num = (dataframe['age'] - 1) / 100
            for idx, row in dataframe.iterrows():
                if row['gender'] == 'f':
                    self.subgroup_indexes[0].append(idx)
                    self.idx2group.append(0)
                else:
                    self.subgroup_indexes[1].append(idx)
                    self.idx2group.append(1)
            subgroup_cnt = [len(g) for g in self.subgroup_indexes]
            self.subgroup_minority = np.argmin(np.array(subgroup_cnt))
            print(f"current dataset minor ratio is {subgroup_cnt[self.subgroup_minority] / sum(subgroup_cnt)}")

        else:
            raise ValueError("no such dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, 'img', self.data.iloc[idx].filenames)).convert('RGB')
        if self.phase == "train":
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, self.target_num[idx]


