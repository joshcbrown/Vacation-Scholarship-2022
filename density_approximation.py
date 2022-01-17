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
from data_structures import MultivariateDataset, theoretical_pdf, theoretical_score
from argparse import ArgumentParser


class SM(nn.Module):
    def __init__(self, n_inputs):
        super(SM, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 30),
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

def keep_grad(output, input, grad_outputs=None):
    return grad(output, input, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=True)[0]
    
def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx

def train(model, optimiser, training_dl, testing_data, cov, mean, args: ArgumentParser):
    for epoch in tqdm(range(args.n_epoch)):
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

        pdf_diff, score_diff = test(model, testing_data, cov, mean, epoch, args.record_results)
        logging.info(f'EPOCH {epoch}\n'
                     f'loss: {loss}\navg pdf diff: {pdf_diff}\n'
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

def main():
    logging.basicConfig(filename='app.log', filemode='w', 
        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-n", "--n-inputs", type=int, default=2, 
                        help="number of inputs to model pdf")
    parser.add_argument("-tr", "--n-training", type=int, default=10000, 
                        help="number of training samples")
    parser.add_argument("-b", "--batch-size", type=int, default=100, 
                        help="size of each mini-batch")
    parser.add_argument("-te", "--n-testing", type=int, default=1000, 
                        help="number of testing samples")
    parser.add_argument("-e", "--n-epoch", type=int, default=100, 
                        help="number of epochs to run the training loop")
    parser.add_argument("-r", "--record-results", action="store_true", default=False, 
                        help="record the results of each epoch in outputs/")
    
    args = parser.parse_args()

    mean = torch.Tensor([0, 0])
    cov = torch.Tensor([[1, 0], [0, 1]])

    training_data = MultivariateDataset(mean, cov, args.n_training, 'data/testing.txt')
    testing_data = MultivariateDataset(mean, cov, args.n_testing, 'data/testing.txt')

    training_dl = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    model = SM(args.n_inputs)
    optimiser = Adam(model.parameters(), lr = 0.5)

    train(model, optimiser, training_dl, testing_data, cov, mean, args)

if __name__ == '__main__':
    main()