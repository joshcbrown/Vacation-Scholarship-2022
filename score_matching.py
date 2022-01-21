import torch
import logging
import numpy as np
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

        if args.activation == "ReLU":
            activation = nn.ReLU
        elif args.activation == "Swish":
            activation = nn.SiLU
            
        self.network = nn.Sequential(
            nn.Linear(args.dim_x, args.hidden_nodes),
            activation(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            activation(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            activation(),
            nn.Linear(args.hidden_nodes, args.hidden_nodes),
            activation(),
            nn.Linear(args.hidden_nodes, args.dim_x if args.mode == 'score' else 1),
        )

    def forward(self, x):
        return self.network(x)

# following two functions adapted from 'Learning the Stein Discrepancy
# for Training and Evaluating Energy-Based Models without Sampling'
def keep_grad(output, input, grad_outputs=None):
    return grad(output, input, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=True)[0]
    
def approx_jacobian_trace(fx, x, M=1):
    tr_dfdx = 0
    for i in range(M):
        eps = torch.randn_like(fx)
        eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
        tr_dfdx += (eps_dfdx * eps).sum(-1)
    
    return tr_dfdx / M

def avg_score_diff(observed, theoretical):
    return ((observed - theoretical) ** 2).sum(1).mean()

def train(model, optimiser, training_dl, testing_data, distribution, args: ArgumentParser):
    for epoch in tqdm(range(args.n_epoch)):
        for i_batch, x in enumerate(training_dl):
            optimiser.zero_grad()
            x.requires_grad_()

            if args.mode == 'density':
                log_qtilde = model(x)
                sq = keep_grad(log_qtilde.sum(), x)
                trH = approx_jacobian_trace(sq, x, args.n_eps)

            elif args.mode == 'score':
                sq = model(x)
                trH = approx_jacobian_trace(sq, x, args.n_eps)
               
            norm_s = (sq * sq).sum(1)
            loss = (trH + .5 * norm_s).mean()

            loss.backward(retain_graph=True)
            optimiser.step()

        if epoch % args.log_frequency == 0:
            te_loss, te_score_diff = test(model, testing_data, distribution, epoch, args)
            tr_theoretical_score = distribution.score_function_vector(x)
            tr_score_diff = avg_score_diff(sq, tr_theoretical_score)
            test_message = (f'EPOCH {epoch}\n'
                            f'training loss: {loss}\n'
                            f'training avg score diff: {tr_score_diff}\n'
                            f'testing loss: {te_loss}\n'
                            f'testing avg score diff: {te_score_diff}\n')
            logging.info(test_message)
            print(test_message)
        
        if args.record_results:
            torch.save(f'outputs/{args.mode}/{epoch}.pt')

def test(model, testing, distribution, epoch, args: ArgumentParser):
    model.eval()
    loss = 0
    nn_score = torch.Tensor()
    actual_score = torch.Tensor()
    for i_batch, x in enumerate(testing):
        x.requires_grad_()
        if args.mode == 'density':
            log_qtilde = model(x)
            sq = keep_grad(log_qtilde.sum(), x)
            trH = approx_jacobian_trace(sq, x, args.n_eps)

        elif args.mode == 'score':
            sq = model(x)
            trH = approx_jacobian_trace(sq, x, args.n_eps)
        
        nn_score = torch.cat((nn_score, sq))
        actual_score = torch.cat((actual_score, distribution.score_function_vector(x)))

        norm_s = (sq * sq).sum(1)
        loss += (trH + .5 * norm_s).mean()

    loss /= i_batch + 1
    score_diff = avg_score_diff(nn_score, actual_score)

    model.train()

    return loss, score_diff

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
    parser.add_argument("-lf", "--log-frequency", type=int, default=1,
                        help="the number of epochs between each log")
    parser.add_argument("-s", "--save", action="store_true", default=False,
                        help="save the model to model.pt")
    parser.add_argument("-a", "--activation", type=str, choices=["ReLU", "Swish"], 
                        default="Swish", help="activation function to use on the net")
    parser.add_argument("-ep", "--n-eps", type=int, default=1, 
                        help="the number of projection vectors used when approximating trace")

    args = parser.parse_args()


    if args.density_model == 'normal':
        mean = torch.zeros(args.dim_x)
        cov = torch.eye(args.dim_x)

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
    testing_dl = DataLoader(testing_data, batch_size=args.batch_size, shuffle=True)

    model = SM(args)
    optimiser = Adam(model.parameters(), lr = 0.001)

    train(model, optimiser, training_dl, testing_dl, distribution, args)
    
    if args.save:
        torch.save(model, 'model.pt')

if __name__ == '__main__':
    main()

# TODO: take the training sample, fit kde, compare avg score diff with that of the model
# try with 2-dimensional x, vary h and plot with existing function
