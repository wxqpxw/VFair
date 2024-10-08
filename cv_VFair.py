import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from resnet import ResNet
import matplotlib.pyplot as plt


class Var(nn.Module):

    def __init__(
            self,
            num_classes=4,
            device='cuda',
            epsilon=1):
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
        self.epsilon = epsilon
        self.softmax = nn.Softmax(dim=1)
        self.E = 0
        self.beta = 0.999
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def learner_step(self, x, targets, regression=True):
        self.frontend.zero_grad()
        self.resnet18.zero_grad()
        self.fc.zero_grad()

        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)

        if not regression:
            loss = F.cross_entropy(logits, targets, reduction="none")
        else:
            loss = F.mse_loss(logits.squeeze(), targets.float(), reduction="none")
        g = torch.mean(loss)
        # calculate global loss or in EMA-way
        with torch.no_grad():
            self.E = self.beta * self.E + (1 - self.beta) * g

        f = torch.sqrt(torch.mean(torch.stack([torch.square(los - self.E) for los in loss])))
        # f = torch.sum(torch.stack([torch.abs(loss[i] - loss[i + 1]) for i in range(len(selected_idx) - 1)]))

        g.backward(retain_graph=True)
        grad_g = self.flatten_grads().detach().clone()

        self.frontend.zero_grad()
        self.resnet18.zero_grad()
        self.fc.zero_grad()
        f.backward()
        grad_f = self.flatten_grads().detach().clone()

        min_loss = torch.min(loss)

        cons1 = self.epsilon - torch.dot(grad_f, grad_g) / torch.square(torch.norm(grad_g))
        cons2 = max(0, (self.E - min_loss) / f)

        lam = max(cons1, cons2)
        grad_final = grad_f + lam * grad_g
        self.assign_grads(grad_final)
        logging_dict = {'g': g, "f": f, "lambda": lam, "cons1": cons1, "cons2": cons2,
                        "EMA_mean_loss": self.E}

        return cons1, cons2, logging_dict

    def flatten_grads(self):
        """
        Flattens the gradients of a model (after `.backward()` call) as a single, large vector.
        :return: 1D torch Tensor
        """
        all_grads = []
        for param in self.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.view(-1))
        return torch.cat(all_grads)

    def assign_grads(self, grads):
        """
        Similar to `assign_weights` but this time, manually assign `grads` vector to a model.
        :param grads: Gradient vectors.
        """
        state_dict = self.state_dict(keep_vars=True)
        index = 0
        for param in state_dict.keys():
            # ignore batchnorm params
            if state_dict[param].grad is None:
                continue
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param].grad = grads[index:index + param_count].view(param_shape).clone()
            index += param_count
        self.load_state_dict(state_dict)
        return

    def learner_test(self, x):
        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)

        pre = torch.argmax(logits, dim=1)
        sof = self.softmax(logits)

        return pre, logits, sof
