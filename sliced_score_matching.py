import torch
import logging
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict as dd
from torch import nn
from torch.autograd import grad
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from score_matching import write_samples, get_samples

class SSM(nn.Module):
    def __init__(self):
        super(SSM, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

def approx_jacobian_trace(fx, x, eps=None):
    if eps is None:
        eps = torch.randn_like(fx)
    
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)

    return tr_dfdx

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=True)[0]

def sliced_J_hat(output, input, eps=None):
    if eps is None:
        eps = torch.randn_like(output)
    return approx_jacobian_trace(output, input, eps) + 1/2 * (output ** 2).sum()

def train(model, optimiser, training, testing, cov, mean, n_epoch=10):
    training.requires_grad = True
    for epoch in tqdm(range(n_epoch)):
        optimiser.zero_grad()
        loss = 0
        eps = torch.randn(2) # MAGIC NUMBER!!
        output = model(training)
        grad(output, training)
        for input in tqdm(training):
            output = model(input)
            loss += sliced_J_hat(output, input, eps)
        loss /= len(training)
        
        loss.backward()
        optimiser.step()

        score_diff = test(model, testing, cov, mean, epoch, True)
        logging.info(f'EPOCH {epoch}:\nloss: {loss}\navg score diff: {score_diff}')

def test(model, testing, cov, mean, epoch, record: bool = False):
    '''compute avg distance between h(x;theta) and theoretical log pdf function'''
    avg_score_diff = 0
    results_dict = dd(list)
    for input in tqdm(testing):
        output = model(input)
        score_diff = abs((output - theoretical_score(input, cov, mean)).sum())
        avg_score_diff += score_diff
        if record:
            results_dict['data_x'].append(input[0].item())
            results_dict['data_y'].append(input[1].item())
            results_dict['score_diff'].append(score_diff.item())
    avg_score_diff /= len(testing)
    if record:
        df = pd.DataFrame.from_dict(results_dict)
        df.to_csv(f'outputs/ssm-{epoch}.csv', index=False)

    return avg_score_diff

def theoretical_score(x, cov, mean):
    return torch.matmul(-cov, (x-mean))

def main():
    logging.basicConfig(filename='app.log', filemode='w', 
        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    mean = torch.Tensor([0, 0])
    cov = torch.Tensor([[1, 0], [0, 1]])
    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
    write_samples(distr)
    training, testing = get_samples()

    model = SSM()
    optimiser = Adam(model.parameters(), lr = 0.001)

    train(model, optimiser, training, testing, cov, mean)


if __name__ == '__main__':
    main()