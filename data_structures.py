from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
import numpy as np

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

if __name__ == '__main__':
    # debugging
    mean = Tensor([0, 0])
    cov = Tensor([[1, 0], [0, 1]])
    data = MultivariateDataset(mean, cov, 10000)
    print(len(data))
    print(data[:5])
