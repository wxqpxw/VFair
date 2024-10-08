import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet import ResNet
import math
import scipy.optimize as sopt


class DRO(nn.Module):
    def __init__(
            self,
            num_classes=4,
            device='gpu',
            phi=0.1
    ):
        super().__init__()

        self.device = device

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet18 = ResNet()
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.phi = phi
        self.C = math.sqrt(1 + (1 / self.phi - 1) ** 2)

    def learner_step(self, x, y, regression=True):
        max_l = 10.
        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)
        if not regression:
            loss = F.cross_entropy(logits, y, reduction="none")
        else:
            loss = F.mse_loss(logits.squeeze(), y.float(), reduction="none")

        foo = lambda eta: self.C * math.sqrt((F.relu(loss - eta) ** 2).mean().item()) + eta
        opt_eta = sopt.brent(foo, brack=(0, max_l))
        loss = self.C * torch.sqrt((F.relu(loss - opt_eta) ** 2).mean()) + opt_eta
        loss.backward()

        logging_dict = {"learner_loss": loss}
        return loss, logits, logging_dict

    def learner_test(self, x):
        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)

        pre = torch.argmax(logits, dim=1)
        sof = self.softmax(logits)

        return pre, logits, sof
