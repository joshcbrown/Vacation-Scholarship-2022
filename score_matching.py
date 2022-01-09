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
    score = grad(output, sample, create_graph=True, retain_graph=True)[0]
    logging.info(score)
    trace = 0
    for i in range(len(score)):
        trace += grad(score[i], sample[i], create_graph=True, retain_graph=True)
    logging.info(trace)
    return 1/2 * (score * score).sum() + trace
    
def approx_jacobian_trace(fx, x):

    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)

    return tr_dfdx

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def train(args, model, train_loader, optimiser, epoch):
    pass

def main():
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    mean = torch.Tensor([1, 2, 3])
    cov = torch.Tensor([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)

    model = SM()
    sample = distr.rsample()
    sample.requires_grad_(True)
    logging.info(sample)
    
    output = model(sample)[0]
    logging.info(output)
    score = keep_grad(output, sample)
    logging.info(score)
    trace = approx_jacobian_trace(score, sample)
    logging.info(trace)
    # logging.info(J_hat(sample, output))
    

if __name__ == '__main__':
    main()