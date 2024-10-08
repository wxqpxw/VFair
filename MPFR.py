import numpy as np
import sklearn
import torch


class MP_Fair_regression:
    '''
    Input:
    x: (n_sample, n_feature) //x contains s.
    s: (n_sample, n_protect_feature)
    y: (n_sample, n_label)
    kernel_xs: kernel function for (x,s)
    kernel_s: kernel function for s
    lmd: regularization parameter
    '''

    def __init__(self, x, s, y, lmd=0, device='cpu'):
        self.x = x
        self.s = s
        self.y = y.to(torch.float32)
        self.n = x.shape[0]
        self.P = None
        self.A = None
        self.device = device

        k = 2 ** s.shape[1]
        self.kernel_s = lambda x, y: sklearn.metrics.pairwise.polynomial_kernel(x, y, degree=k, gamma=None, coef0=1)
        self.kernel_xs = lambda x, y: sklearn.metrics.pairwise.rbf_kernel(x, y, gamma=0.1)

        self.K = torch.tensor(self.kernel_xs(self.x, self.x), dtype=torch.float32).to(device)
        self.K_s = torch.tensor(self.kernel_s(self.s, self.s), dtype=torch.float32).to(device)
        self.lmd = lmd

    def fit(self):

        # Centralized Kernel Matrix
        H = torch.eye(self.n, device=self.device) - 1 / self.n * torch.ones((self.n, self.n), device=self.device)

        K_b = torch.matmul(H, torch.matmul(self.K, H))
        K_sb = torch.matmul(H, torch.matmul(self.K_s, H))

        # Eigenvector Computation
        K_eigen = torch.matmul(K_sb, K_b)
        self.m = torch.linalg.matrix_rank(K_eigen)

        eigvals, eigvecs = torch.linalg.eig(K_eigen)
        A = eigvecs[:, 0:self.m].real.to(self.device)

        # Uncentralization
        A = torch.matmul(H, A)

        # Gram-Schmidt Process
        for i in range(self.m):
            a_i = A[:, i:i+1]
            for j in range(i):
                a_j = A[:, j:j+1]
                a_i = a_i - a_j * (a_j.t().matmul(self.K).matmul(a_i))
            # Normalization
            a_i = a_i / torch.sqrt(a_i.t().matmul(self.K).matmul(a_i))
            A[:, i:i+1] = a_i

        # # The off-diagonal elements should be 0 and diagonal elements should be 1
        # print(A.T.dot(self.K.dot(A)))

        # Projection
        P = torch.eye(self.n, device=self.device) - A.matmul(A.t()).matmul(self.K)

        self.w_ = P.t().matmul(self.K).matmul(self.y)
        self.w_ = torch.pinverse(P.t().matmul(self.K).matmul(self.K).matmul(P)
                                 + self.lmd * P.t().matmul(self.K).matmul(P)).matmul(self.w_)
        self.w_ = P.matmul(self.w_)
        self.A = A
        self.P = P
        if self.kernel_xs.__name__ == 'linear_kernel':
            self.w_ = self.x.t().matmul(self.w_)
        return self.w_

    def pred(self, x_):
        if self.kernel_xs.__name__=='linear_kernel':
            y_ = x_.matmul(self.w_)
        else:
            y_ = torch.tensor(self.kernel_xs(x_.to(torch.float32), self.x.to(torch.float32))).matmul(self.w_)
        return y_

    # def fit(self):
    #
    #     # Centralized Kernel Matrix
    #     H = torch.tensor(np.eye(self.n)-1/self.n*np.ones((self.n, self.n))).to(self.device)
    #
    #     K_b = H.dot(self.K.dot(H))
    #     K_sb = H.dot(self.K_s.dot(H))
    #
    #     # Eigenvector Computation
    #     K_eigen = K_sb.dot(K_b)
    #     self.m=np.linalg.matrix_rank(K_eigen)
    #     # print('New m: ', self.m)
    #
    #     eigvals, eigvecs = np.linalg.eig(K_eigen)
    #     A = eigvecs[:, 0:self.m].real
    #
    #     # Uncentralization
    #     A = H.dot(A)
    #
    #     # Gram-Schmidt Process
    #     for i in range(self.m):
    #         a_i = A[:, i:i+1]
    #         for j in range(i):
    #             a_j = A[:, j:j+1]
    #             a_i = a_i-a_j*(a_j.T.dot(self.K).dot(a_i))
    #         # Normalization
    #         a_i = a_i/np.sqrt(a_i.T.dot(self.K).dot(a_i))
    #         A[:, i:i+1] = a_i
    #
    #     # # The off-diagonal elements should be 0 and diagonal elements should be 1
    #     # print(A.T.dot(self.K.dot(A)))
    #
    #     # Projection
    #     P = np.eye(self.n)-A.dot(A.T).dot(self.K)
    #
    #     self.w_ = P.T.dot(self.K).dot(self.y)
    #     self.w_ = np.linalg.pinv(P.T.dot(self.K).dot(self.K).dot(P)+self.lmd*P.T.dot(self.K).dot(P)).dot(self.w_)
    #     self.w_ = P.dot(self.w_)
    #     self.A=A
    #     self.P=P
    #     if self.kernel_xs.__name__=='linear_kernel':
    #         self.w_=self.x.T.dot(self.w_)
    #     return self.w_
    #
    # def pred(self, x_):
    #     if self.kernel_xs.__name__=='linear_kernel':
    #         y_=x_.dot(self.w_)
    #     else:
    #         y_ =  self.kernel_xs(x_, self.x).dot(self.w_)
    #     return y_

