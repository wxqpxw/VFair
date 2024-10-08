import numpy as np
import sklearn
import torch


class Fair_kernel_learning:
    '''
    Fair Kernel Learning (https://arxiv.org/pdf/1710.05578.pdf): a regularizer-based method aims to eliminate the covariance between the predicted value and the sensitive attributes.
    The implementation is borrowed from https://isp.uv.es/soft_regression.html.

    Input:
    x: (n_sample, n_feature) //x contains s.
    s: (n_sample, n_protect_feature)
    y: (n_sample, n_label)
    kernel_xs: kernel function for (x,s)
    lmd: regularization parameter
    eta: penalty parameter
    '''

    def __init__(self, x, s, y, lmd=0, eta=0, device='cpu'):
        self.x = x
        self.s = s
        self.y = y.to(torch.float32)
        self.n = x.shape[0]
        self.device = device

        self.kernel_xs = lambda x, y: sklearn.metrics.pairwise.rbf_kernel(x, y, gamma=0.1)

        self.K = torch.tensor(self.kernel_xs(self.x, self.x), dtype=torch.float32).to(device)
        self.K_s = torch.tensor(self.kernel_xs(self.s, self.s), dtype=torch.float32).to(device)
        self.lmd = lmd
        self.eta = eta

    def fit(self):
        # Centralized Kernel Matrix
        H = torch.eye(self.n, device=self.device) - 1 / self.n * torch.ones((self.n, self.n), device=self.device)
        K_sb = torch.matmul(H, torch.matmul(self.K_s, H))
        # wd  = (la*eye(ntr) + K + mus(k)*HKqH*K)\(ytr);
        self.w_ = torch.linalg.pinv(self.K + self.lmd * torch.eye(self.n, device=self.device) +
                                    self.eta / self.n ** 2 * K_sb.matmul(self.K)).matmul(self.y)
        if self.kernel_xs.__name__ == 'linear_kernel':
            self.w_ = self.x.t().matmul(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__ == 'linear_kernel':
            y_ = x_.matmul(self.w_)
        else:
            y_ = torch.tensor(self.kernel_xs(x_.to(torch.float32), self.x.to(torch.float32))).matmul(self.w_)
        return y_
