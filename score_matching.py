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
from data_structures import *
from argparse import ArgumentParser


class SM(nn.Module):
    def __init__(self, args):
        super(SM, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(args.dim_x, args.hidden_nodes),
            Swish(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            Swish(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            Swish(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            Swish(),
            nn.Linear(args.hidden_nodes, args.dim_x if args.mode == 'score' else 1),
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

def train(model, optimiser, training_dl, testing_data, distribution, args: ArgumentParser):
    for epoch in tqdm(range(args.n_epoch)):
        for i_batch, x in enumerate(training_dl):
            optimiser.zero_grad()
            x.requires_grad_()

            if args.mode == 'density':
                log_qtilde = model(x)
                sq = keep_grad(log_qtilde.sum(), x)
                trH = approx_jacobian_trace(sq, x)

            elif args.mode == 'score':
                sq = model(x)
                trH = approx_jacobian_trace(sq, x)
                
            norm_s = (sq * sq).sum(1)
            loss = (trH + .5 * norm_s).mean()

            loss.backward(retain_graph=True)
            optimiser.step()

        if epoch % args.log_frequency == 0:
            pdf_diff, score_diff = test(model, testing_data, distribution, epoch, args)
            if args.mode == 'density':
                logging.info(f'EPOCH {epoch}\n'
                            f'loss: {loss}\navg' # pdf diff: {pdf_diff}\n'
                            f'avg score diff: {score_diff}')
            elif args.mode == 'score':
                logging.info(f'EPOCH {epoch}:\nloss: {loss}\navg score diff: {score_diff}')

def test(model, testing, distribution, epoch, args: ArgumentParser):
    '''compute avg distance between h(x;theta) and empirical log pdf function'''
    avg_pdf_diff = 0
    avg_score_diff = 0
    results_dict = dd(list)
    for input in tqdm(testing):
        if args.mode == 'score':
            nn_score = model(input)
        elif args.mode == 'density':
            output = model(input)
            nn_score = keep_grad(output, input)

        score_diff = abs((nn_score - distribution.score_function(input)).sum())
        avg_score_diff += score_diff

        if args.mode == 'density':
            # not actually a helpful metric since normalising constant is probably wrong?
            # pdf_diff = abs(np.exp(output.detach()) - math.log(distribution.pdf(input)))[0]
            # avg_pdf_diff += pdf_diff
            pass

        if args.record_results:
            results_dict['data_x'].append(input[0].item())
            results_dict['data_y'].append(input[1].item())
            results_dict['score_diff'].append(score_diff.item())
            if args.mode == 'density':
                # results_dict['pdf_diff'].append(pdf_diff.item())
                results_dict['log_q'].append(output.item())

    if args.mode == 'density':
        # avg_pdf_diff /= len(testing)
        pass
    avg_score_diff /= len(testing)

    if args.record_results:
        df = pd.DataFrame.from_dict(results_dict)
        df.to_csv(f'outputs/{args.mode}/{epoch}.csv', index=False)

    return avg_pdf_diff, avg_score_diff

def main():
    logging.basicConfig(filename='app.log', filemode='w', 
        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["density", "score"], 
                        default="density", help="which function the model will approximate")
    parser.add_argument("-d", "--density-model", type=str, choices=["normal", "gbrbm"], 
                        default="normal", help="the model which the neural net will approximate")
    parser.add_argument("-hn", "--hidden-nodes", type=int, default=30, 
                        help="number of hidden nodes in each hidden layer")
    parser.add_argument("-dx", "--dim-x", type=int, default=2, 
                        help="number of inputs to model pdf")
    parser.add_argument("-dh", "--dim-h", type=int, default=0, 
                        help="dimensions of parameter h (only for GBRBM)")
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
    parser.add_argument("-lf", "--log-frequency", type=int, default=2,
                        help="the amount of epochs between each log")    

    args = parser.parse_args()


    if args.density_model == 'normal':
        mean = torch.Tensor([0, 0])
        cov = torch.Tensor([[1, 0], [0, 1]])

        distribution = MultivariateNormal2(mean, cov)

        training_data = MultivariateDataset(distribution, args.n_training, 'data/testing.txt')
        testing_data = MultivariateDataset(distribution, args.n_testing, 'data/testing.txt')
    elif args.density_model == 'gbrbm':
        B = randb((args.dim_x, args.dim_h)) * 2. - 1.
        c = torch.randn((1, args.dim_h))
        b = torch.randn((1, args.dim_x))

        distribution = GaussianBernoulliRBM(B, b, c)

        training_data = GBRMBDataset(distribution, args.n_training)
        testing_data = GBRMBDataset(distribution, args.n_testing)

    training_dl = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    model = SM(args)
    optimiser = Adam(model.parameters(), lr = 0.001)

    train(model, optimiser, training_dl, testing_data, distribution, args)

if __name__ == '__main__':
    main()
