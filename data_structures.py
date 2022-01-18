import torch
import math
import numpy as np
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from torch import nn

class MultivariateDataset(Dataset):
    def __init__(self, loc: Tensor, cov: Tensor, length: int, file_out: str = None):
        self.distribution = MultivariateNormal(loc, cov)
        self.sample = self.distribution.rsample((1, length))[0]
        self.sample.requires_grad_()
        if file_out is not None:
            np.savetxt(file_out, self.sample.detach().numpy())

    def __len__(self):
        return self.sample.shape[0]

    def __getitem__(self, index: int):
        return self.sample[index]

class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))
    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)

def theoretical_pdf(x, cov, mean):
    Z_theta = math.sqrt(torch.det(2*math.pi*cov))
    q_x = math.exp(-1/2 * torch.matmul(torch.matmul(x-mean, cov), x-mean))
    return q_x / Z_theta

def theoretical_score(x, cov, mean):
    return torch.matmul(-cov, (x-mean))

if __name__ == '__main__':
    # debugging
    a = torch.randn(5)
    model = Swish()
    b = model(a)
    print(b)
