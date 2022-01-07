import torch
import logging
from torch import nn
from torch.autograd import grad
from torch.distributions.multivariate_normal import MultivariateNormal

class SM(nn.Module):
    def __init__(self):
        super(SM, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

    def get_grads(self):
        pass
        

def J_hat(sample: torch.Tensor, output: torch.Tensor) -> float:
    score = grad(output, sample, create_graph=True)[0]
    logging.info(score)
    trace = 0
    for i in range(len(score)):
        trace += grad(score[i], sample[i])
    logging.info(trace)
    return 1/2 * (score * score).sum() + trace
    

def train(args, model, train_loader, optimiser, epoch):
    pass

def main():
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    mean = torch.Tensor([1, 2, 3])
    cov = torch.Tensor([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)

    model = SM()
    sample = distr.rsample()
    sample.requires_grad = True
    logging.info(sample)
    
    output = model(sample)[0]
    logging.info(output)
    logging.info(J_hat(sample, output))
    

if __name__ == '__main__':
    main()