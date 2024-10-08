import torch
import torch.nn as nn
import torch.nn.functional as F
from ERM import baselineNN
import math
import scipy.optimize as sopt


class DRO(nn.Module):

    def __init__(
            self,
            embedding_size,
            n_num_cols,
            learner_hidden_units=[64, 32],
            activation_fn=nn.ReLU,
            device='cpu',
            phi=0.1,
            train_dataset=None):
        super().__init__()

        self.device = device

        self.learner = baselineNN(
            embedding_size,
            n_num_cols,
            learner_hidden_units,
            activation_fn=activation_fn,
            device=device
        )
        self.phi = phi
        self.learner.to(device)
        self.C = math.sqrt(1 + (1 / self.phi - 1) ** 2)

    def learner_step(self, x_cat, x_num, targets, regression=True):
        max_l = 10.
        self.learner.zero_grad()
        logits, _, _ = self.learner(x_cat, x_num)
        if regression:
            loss = F.mse_loss(logits, targets, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        foo = lambda eta: self.C * math.sqrt((F.relu(loss - eta) ** 2).mean().item()) + eta
        opt_eta = sopt.brent(foo, brack=(0, max_l))
        loss = self.C * torch.sqrt((F.relu(loss - opt_eta) ** 2).mean()) + opt_eta
        loss.backward()
        logging_dict = {"learner_loss": loss}
        return loss, logits, logging_dict
