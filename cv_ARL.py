import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet


class AdversaryNN(nn.Module):
    def __init__(self, n_hidden, device='cuda'):
        super().__init__()
        self.device = device

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.hidden = nn.Linear(64 * 50 * 42, n_hidden)
        self.last = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The forward step for the adversary.
        """
        x = self.frontend(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.last(x)

        x = self.sigmoid(x)
        x_mean = torch.mean(x)
        x = x / torch.max(torch.Tensor([x_mean, 1e-4]))
        x = x + torch.ones_like(x)

        return x


class ARL(nn.Module):

    def __init__(
            self,
            num_classes=4,
            n_hidden=128,
            batch_size=256,
            device='cuda',
    ):
        super().__init__()

        self.device = device
        self.adversary_weights = torch.ones(batch_size, 1)

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet18 = ResNet()
        self.adversary = AdversaryNN(
           n_hidden, device=device
        )
        self.fc = nn.Linear(512, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.softmax = nn.Softmax(dim=1)

        self.adversary.to(device)

    def learner_step(self, x, y, regression=True):
        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)

        adversary_weights = self.adversary_weights.to(self.device)
        loss = self.get_learner_loss(logits, y, adversary_weights, regression)
        loss.backward()

        # Predictions are returned to trainer for fairness metrics
        logging_dict = {"learner_loss": loss}
        return loss, logits, logging_dict

    def adversary_step(self, x, learner_logits, targets, regression=True):
        """
        Performs one loop
        """
        self.adversary.zero_grad()

        adversary_weights = self.adversary(x)
        self.adversary_weights = adversary_weights.detach()

        loss = self.get_adversary_loss(
            learner_logits.detach(), targets, adversary_weights
        )

        loss.backward()

        logging_dict = {"adv_loss": loss}
        return loss, logging_dict

    def get_learner_loss(self, logits, targets, adversary_weights, regression=True):
        """
        Compute the loss for the learner.
        """
        if not regression:
            loss = F.cross_entropy(logits, targets, reduction="none")
        else:
            loss = F.mse_loss(logits.squeeze(), targets.float(), reduction="none")
        weighted_loss = torch.mul(adversary_weights.squeeze(), loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, logits, targets, adversary_weights):
        """
        Compute the loss for the adversary.
        """
        loss = F.cross_entropy(logits, targets, reduction="none")
        weighted_loss = torch.mul(-adversary_weights, loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def learner_test(self, x):
        x = self.frontend(x)
        x = self.resnet18(x)
        logits = self.fc(x)

        pre = torch.argmax(logits, dim=1)
        sof = self.softmax(logits)

        return pre, logits, sof
