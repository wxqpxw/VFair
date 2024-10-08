import torch
import numpy as np
import json
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score


class FairnessMetrics:
    def __init__(self, averaged_over, subgroup_idx, is_regression=True):
        """
        Implements evaluation metrics for measuring performance of the VFair model.

        Args:
            averaged_over: the amount of iterations the model is ran (for averaging 
                     the results).
        """
        self.logging_dict = {"utility": 0}
        self.utility = [[] for i in range(averaged_over)]
        self.subgroup_indexes = subgroup_idx
        self.var = [[] for i in range(averaged_over)]
        self.worst = [[] for i in range(averaged_over)]
        self.diff = [[] for i in range(averaged_over)]
        self.sum = [[] for i in range(averaged_over)]
        self.regression = is_regression
        self.average_over = averaged_over

    def set_utility(self, pred, targets, n_iter):
        """
        Calculates the accuracy score.

        Args:
            pred: prediction (Torch tensor).
            targets: target variables (Torch tensor).
            n_iter: iteration of this training loop. 
        """
        if self.regression:
            utility = mean_squared_error(targets.cpu().detach().numpy(), pred.cpu().detach().numpy())
        else:
            utility = accuracy_score(targets.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")
        self.utility[n_iter].append(utility)
        self.logging_dict["utility"] = utility
        return utility

    def average_results(self):
        """
        Averages the results of all iterations.
        """
        self.utility_avg = np.mean(np.array(self.utility), axis=0)
        self.var_avg = np.mean(np.array(self.var), axis=0)
        self.worst_avg = np.mean(np.array(self.worst), axis=0)
        self.diff_avg = np.mean(np.array(self.diff), axis=0)
        self.sum_avg = np.mean(np.array(self.sum), axis=0)

        if self.average_over == 1:
            self.utility_var = [0]
            self.worst_var = [0]
            self.diff_var = [0]
            self.sum_var = [0]
            self.var_var = [0]
        else:
            self.utility_var = np.sqrt(np.var(self.utility, axis=0))
            self.worst_var = np.sqrt(np.var(self.worst, axis=0))
            self.diff_var = np.sqrt(np.var(self.diff, axis=0))
            self.sum_var = np.sqrt(np.var(self.sum, axis=0))
            self.var_var = np.sqrt(np.var(self.var, axis=0))

    def save_metrics(self, res_dir):
        """
        Saves the averaged metrics in a json file.
        """

        metrics = {
            "utility_avg": float(self.utility_avg[-1]),
            "worst_avg": float(self.worst_avg[-1]),
            "diff_avg": float(self.diff_avg[-1]),
            "sum_avg": float(self.sum_avg[-1]),
            "var_avg": float(self.var_avg[-1]),
            "utility_var": float(self.utility_var[-1]),
            "worst_var": float(self.worst_var[-1]),
            "diff_var": float(self.diff_var[-1]),
            "sum_var": float(self.sum_var[-1]),
            "var_var": float(self.var_var[-1]),
        }
        json.dump(metrics, open("{}.json".format(res_dir), 'w'))

    def set_var(self, pred, targets, n_iters):
        # logits
        if self.regression:
            loss = F.mse_loss(pred.squeeze(), targets.squeeze(), reduction='none')
        else:
            if targets.size() == pred.size():
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(), targets, reduction="none")
            else:
                loss = F.cross_entropy(pred, targets, reduction="none")
        variance = torch.var(loss).cpu().detach().numpy()
        self.var[n_iters].append(variance)
        self.logging_dict["var"] = variance

    def set_utility_other(self, pred, targets, n_iter):
        utilities = []
        for group_idx in self.subgroup_indexes:
            group_pred = pred[group_idx].cpu().detach().numpy()
            group_tar = targets[group_idx].cpu().detach().numpy()
            if self.regression:
                group_utility = torch.tensor(mean_squared_error(group_tar, group_pred))
            else:
                group_utility = torch.tensor(accuracy_score(group_tar, group_pred))
                # group_utility = torch.tensor(f1_score(group_tar, group_pred, average="macro"))
            utilities.append(group_utility)
        utilities = torch.tensor(utilities)
        # here use global mean
        miu = self.utility[n_iter][-1]
        tad = torch.tensor([torch.abs(utility - miu) for utility in utilities])
        tad = torch.sum(tad)
        if self.regression:
            self.worst[n_iter].append(torch.max(utilities))
        else:
            self.worst[n_iter].append(torch.min(utilities))
        self.diff[n_iter].append(torch.max(utilities) - torch.min(utilities))
        self.sum[n_iter].append(tad)

        self.logging_dict["worst"] = torch.max(utilities)
        self.logging_dict["diff"] = torch.max(utilities) - torch.min(utilities)
        self.logging_dict["sum"] = tad
        return

    def set_error_other(self, pred, targets, n_iter):
        errors = []
        for group_idx in self.subgroup_indexes:
            group_pred = pred[group_idx].cpu().detach().numpy()
            group_tar = targets[group_idx].cpu().detach().numpy()
            group_error = torch.tensor(mean_squared_error(group_tar, group_pred))
            errors.append(group_error)
        errors = torch.tensor(errors)
        # here use global mean
        miu = mean_squared_error(targets, pred)
        tad = torch.tensor([torch.abs(error - miu) for error in errors])
        tad = torch.sum(tad)
        self.worst[n_iter].append(torch.max(errors))
        self.diff[n_iter].append(torch.max(errors) - torch.min(errors))
        self.sum[n_iter].append(tad)
        self.utility[n_iter].append(miu)
        self.logging_dict["utility"] = miu
        self.logging_dict["worst"] = torch.max(errors)
        self.logging_dict["diff"] = torch.max(errors) - torch.min(errors)
        self.logging_dict["sum"] = tad
        return
