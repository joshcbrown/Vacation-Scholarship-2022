import torch
import torch.distributions as distributions
import math
import numpy as np
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from torch import nn
from tqdm import tqdm

class MultivariateNormal2(MultivariateNormal):
    def __init__(self, loc, cov):
        super(MultivariateNormal2, self).__init__(loc, cov)
        self.loc = loc
        self.cov = cov
    
    def pdf(self, x):
        mean = self.loc
        cov = self.cov
        Z_theta = math.sqrt(torch.det(2*math.pi*cov))
        q_x = math.exp(-1/2 * torch.matmul(torch.matmul(x-mean, cov), x-mean))
        return q_x / Z_theta

    def score_function(self, x):
        return -self.cov @ (x - self.mean)

    def score_function_vector(self, x):
        """equivalent to score_function, but instead computes for a vector of points
        rather than a single point"""
        return (-self.cov @ (x - self.mean).transpose(0, 1)).transpose(0, 1)

class MultivariateDataset(Dataset):
    def __init__(self, distribution: MultivariateNormal2, length: int, file_out: str = None):
        self.distribution = distribution
        self.sample = self.distribution.rsample((1, length))[0]
        self.sample.requires_grad_()
        if file_out is not None:
            np.savetxt(file_out, self.sample.detach().numpy())

    def __len__(self):
        return self.sample.shape[0]

    def __getitem__(self, index: int):
        return self.sample[index]


# code from 'Learning the Stein Discrepancy
# for Training and Evaluating Energy-Based Models without Sampling'
def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()

class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        # self.B = B
        # self.b = b
        # self.c = c
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in

    def score_function(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in tqdm(range(self.burn_in)):
            x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h

class GBRMBDataset(Dataset):
    def __init__(self, distribution: GaussianBernoulliRBM, length):
        self.distribution = distribution
        self.sample = distribution.sample(length)

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, index: int):
        return self.sample[index]


if __name__ == '__main__':
    # debugging
    pass
