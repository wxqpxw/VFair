import torch
import torch.nn as nn
import torch.nn.functional as F
from ERM import baselineNN


class VFair(nn.Module):

    def __init__(
            self,
            embedding_size,
            n_num_cols,
            learner_hidden_units=[64, 32],
            activation_fn=nn.ReLU,
            device='cpu',
            epsilon=1,
            train_dataset=None):
        super().__init__()

        self.device = device

        self.learner = baselineNN(
            embedding_size=embedding_size,
            n_num_cols=n_num_cols,
            n_hidden=learner_hidden_units,
            activation_fn=activation_fn,
            device=device
        )
        self.learner.to(device)
        self.global_data = train_dataset
        self.epsilon = epsilon

    def learner_step(self, x_cat, x_num, targets, regression=True):
        self.learner.zero_grad()
        logits, sig, _ = self.learner(x_cat, x_num)

        loss = F.binary_cross_entropy_with_logits(logits[selected_idx], targets[selected_idx], reduction='none')

        if regression:
            loss = F.mse_loss(logits, targets, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        g = torch.mean(loss)

        g_cat, g_num, g_tar = self.global_data[:]
        with torch.no_grad():
            if g_cat is not None:
                g_cat = g_cat.to(self.device)
            g_log, _, _ = self.learner(g_cat, g_num.to(self.device).to(torch.float32))
            if regression:
                mean_loss = F.mse_loss(g_log.squeeze(), g_tar.to(self.device), reduction="mean")
            else:
                mean_loss = F.binary_cross_entropy_with_logits(g_log.squeeze(), g_tar.to(self.device), reduction="mean")

        f = torch.sqrt(torch.mean(torch.stack([torch.square(los - mean_loss) for los in loss])))

        g.backward(retain_graph=True)
        grad_g = self.flatten_grads().detach().clone()

        self.learner.zero_grad()
        f.backward()
        grad_f = self.flatten_grads().detach().clone()

        min_loss = torch.min(loss)

        cons1 = self.epsilon - torch.dot(grad_f, grad_g) / torch.square(torch.norm(grad_g))
        cons2 = max(0, (mean_loss - min_loss) / f)

        lam = max(cons1, cons2)
        grad_final = grad_f + lam * grad_g
        self.assign_grads(grad_final)
        logging_dict = {
            'cons1': cons1,
            'cons2': cons2,
            'g': g,
            'f': f,
            'lam': lam
        }
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
