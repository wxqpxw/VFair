import random

import numpy as np

from argparser import get_args
from ARL import ARL
from DRO import DRO
from ERM import ERM
from dataloader import loadDataset
from metrics import FairnessMetrics
from VFair import VFair
import torch


def test():
    for dataset in ['uci_adult', 'law_school', 'compas']:
        test_dataset = loadDataset(dataset=dataset, train_or_test="test")
        _, _, test_target = test_dataset[:]
        data_len = len(test_dataset)
        # wacc mad tad acc
        disparity = {
            "baseline": [[], [], [], []],
            "ARL": [[], [], [], []],
            "var": [[], [], [], []]
        }
        for _ in range(10):  # random 10 * 10 .pth = random 100
            split = 20
            sub_ratio = [random.random() for i in range(split)]
            regular = sum(sub_ratio)
            sub_ratio = [sub_rat / regular for sub_rat in sub_ratio]
            sub_len = [max(1, int(sub_rat * data_len)) for sub_rat in sub_ratio]
            sub_len[split - 1] = data_len - sum(sub_len[:split - 1])

            indices = list(range(data_len))
            random.shuffle(indices)
            subgroup_indexes = []
            start = 0
            for sub_l in sub_len:
                subgroup_indexes.append(indices[start: start + sub_l])
                start += sub_l
            subgroup_minority = np.argmin(sub_len)
            for method in ['baseline', 'ARL', 'var']:
                metrics = FairnessMetrics(1, subgroup_indexes, subgroup_minority)
                sigs = torch.load(f'checkpoints/final/final/{dataset}/{method}/sigs')
                preds = torch.load(f'checkpoints/final/final/{dataset}/{method}/preds')
                for i in range(10):
                    metrics.set_error_other(sigs[i], test_target, 0)
                disparity[method][0].extend(metrics.worst[0])
                disparity[method][1].extend(metrics.diff[0])
                disparity[method][2].extend(metrics.sum[0])
                disparity[method][3].extend(metrics.utility[0])

        worst = [0, 0, 0, 0]
        diff = [0, 0, 0, 0]
        tad = [0, 0, 0, 0]
        acc = [0, 0, 0, 0]
        for i in range(100):
            worst_ranked = np.argsort(
                [disparity["baseline"][0][i], disparity["ARL"][0][i], disparity["var"][0][i]])
            diff_ranked = np.argsort(
                [disparity["baseline"][1][i], disparity["ARL"][1][i], disparity["var"][1][i]])
            sum_ranked = np.argsort(
                [disparity["baseline"][2][i], disparity["ARL"][2][i], disparity["var"][2][i]])
            acc_ranked = np.argsort(
                [disparity["baseline"][3][i], disparity["ARL"][3][i], disparity["var"][3][i]])

            for j in range(3):
                worst[worst_ranked[j]] += 3 - j
                diff[diff_ranked[j]] += j + 1
                tad[sum_ranked[j]] += j + 1
                acc[acc_ranked[j]] += 3 - j
        worst = [wor / 100 for wor in worst]
        diff = [dif / 100 for dif in diff]
        tad = [ta / 100 for ta in tad]
        acc = [ac / 100 for ac in acc]
        print(f"{dataset} ERM ARL VFAIR")
        print(f"Utility: {acc[0]} {acc[1]} {acc[2]}")
        print(f"WU: {worst[0]} {worst[1]} {worst[2]}")
        print(f"MAD: {diff[0]} {diff[1]} {diff[2]}")
        print(f"TAD: {tad[0]} {tad[1]} {tad[2]}")


def generate_preds():
    for dataset in ['uci_adult', 'law_school', 'compas']:
        train_dataset = loadDataset(
            dataset=dataset,
            train_or_test="train",
            embedding_size=args.embedding_size,
        )
        test_dataset = loadDataset(dataset=dataset, train_or_test="test")
        for method in ['baseline', 'ARL', 'var']:
            if method == "ARL":
                model = ARL(embedding_size=train_dataset.categorical_embedding_sizes,
                            n_num_cols=len(train_dataset.mean_std.keys()))
            elif method == "baseline":
                model = ERM(embedding_size=train_dataset.categorical_embedding_sizes,
                            n_num_cols=len(train_dataset.mean_std.keys()))
            elif method == "var":
                model = VFair(embedding_size=train_dataset.categorical_embedding_sizes,
                              n_num_cols=len(train_dataset.mean_std.keys()))
            elif method == "DRO":
                model = DRO(embedding_size=train_dataset.categorical_embedding_sizes,
                            n_num_cols=len(train_dataset.mean_std.keys()))
            else:
                print("Unknown model")
            preds = []
            for i in range(10):
                model.load_state_dict(
                    torch.load(f'checkpoints/{dataset}/{method}/checkpoints/model_checkpoint{i}.pt',
                               map_location='cpu'))
                test_cat, test_num, test_target = test_dataset[:]
                with torch.no_grad():
                    test_logits, test_sigmoid, test_pred = model.learner(
                        test_cat, test_num
                    )
                preds.append(test_pred)
            torch.save(preds, f'final/{dataset}/{method}/preds')


if __name__ == '__main__':
    args = get_args()
    generate_preds()
    test()
