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
from torch.utils.data import DataLoader
from data_structures import MultivariateDataset


class SM(nn.Module):
    def __init__(self):
        super(SM, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 30),
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

def sliced_J_hat(output, sample):
    score = keep_grad(output, sample)
    trace, eps = approx_jacobian_trace(score, sample)
    return 1/2 * ((eps * score).sum()) ** 2 + trace

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=True)[0]


def train(model, optimiser, training_dl, testing_data, cov, mean, n_epoch=100):
    for epoch in tqdm(range(n_epoch)):
        for i_batch, x in enumerate(training_dl):
            optimiser.zero_grad()
            x.requires_grad_()

            log_qtilde = model(x)
            sq = keep_grad(log_qtilde.sum(), x)

            trH = approx_jacobian_trace(sq, x)
            norm_s = (sq * sq).sum(1)
            loss = (trH + .5 * norm_s).mean()
            loss.backward()
            optimiser.step()
        pdf_diff, score_diff = test(model, testing_data, cov, mean, epoch, False)
        logging.info(f'EPOCH {epoch}\n'
                     f'loss {loss}\navg pdf diff: {pdf_diff}\n'
                     f'avg score diff: {score_diff}')
            
def test(model, testing, cov, mean, epoch, record: bool = False):
    '''compute avg distance between h(x;theta) and empirical log pdf function'''
    avg_pdf_diff = 0
    avg_score_diff = 0
    results_dict = dd(list)
    for input in tqdm(testing):
        output = model(input)
        nn_score = keep_grad(output, input)
        score_diff = abs((nn_score - theoretical_score(input, cov, mean)).sum())
        pdf_diff = abs(output - math.log(theoretical_pdf(input, cov, mean)))[0]
        avg_score_diff += score_diff
        avg_pdf_diff += pdf_diff
        if record:
            results_dict['data_x'].append(input[0].item())
            results_dict['data_y'].append(input[1].item())
            results_dict['score_diff'].append(score_diff.item())
            results_dict['pdf_diff'].append(pdf_diff.item())
    avg_pdf_diff /= len(testing)
    avg_score_diff /= len(testing)
    if record:
        df = pd.DataFrame.from_dict(results_dict)
        df.to_csv(f'outputs/{epoch}.csv', index=False)

    return avg_pdf_diff, avg_score_diff

def theoretical_pdf(x, cov, mean):
    Z_theta = math.sqrt(torch.det(2*math.pi*cov))
    q_x = math.exp(-1/2 * torch.matmul(torch.matmul(x-mean, cov), x-mean))
    return q_x / Z_theta

def theoretical_score(x, cov, mean):
    return torch.matmul(-cov, (x-mean))

def write_samples(distribution, train_size=10000, test_size=1000):
    training = distribution.rsample((1, train_size))[0]
    testing = distribution.rsample((1, test_size))[0]
    np.savetxt('data/training.txt', training.numpy())
    np.savetxt('data/testing.txt', testing.numpy())

def get_samples():
    training = torch.Tensor(np.loadtxt('data/training.txt'))
    training.requires_grad_(True)
    testing = torch.Tensor(np.loadtxt('data/testing.txt'))
    testing.requires_grad_(True)
    return training, testing

def main():
    logging.basicConfig(filename='app.log', filemode='w', 
        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    mean = torch.Tensor([0, 0])
    cov = torch.Tensor([[1, 0], [0, 1]])
    training_data = MultivariateDataset(mean, cov, 10000, 'data/testing.txt')
    testing_data = MultivariateDataset(mean, cov, 1000, 'data/testing.txt')

    training_dl = DataLoader(training_data, batch_size=100, shuffle=True)

    model = SM()
    optimiser = Adam(model.parameters(), lr = 0.5)

    train(model, optimiser, training_dl, testing_data, cov, mean)

if __name__ == '__main__':
    main()