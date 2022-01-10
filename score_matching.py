import torch
import logging
import numpy as np
import math
from tqdm import tqdm
from torch import nn
from torch.autograd import grad
from torch.optim import Adam
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

    return tr_dfdx, eps

def sliced_J_hat(output, sample):
    score = keep_grad(output, sample)
    trace, eps = approx_jacobian_trace(score, sample)
    return 1/2 * ((eps * score).sum()) ** 2 + trace

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=True)[0]


def train(model, optimiser, training, testing, cov, mean, n_epoch=5):
    training.requires_grad = True
    for epoch in tqdm(range(n_epoch)):
        optimiser.zero_grad()
        loss = 0

        for input in tqdm(training):
            output = model(input)
            loss += sliced_J_hat(output[0], input)
        loss /= len(training)

        logging.info(f'loss {epoch}: {loss}')
        loss.backward()
        optimiser.step()

        test(model, testing, cov, mean)
    
def test(model, testing, cov, mean):
    '''compute avg distance between h(x;theta) and empirical log pdf function'''
    avg_pdf_diff = 0
    avg_score_diff = 0
    for input in tqdm(testing):
        output = model(input)
        nn_score = keep_grad(output, input)
        avg_score_diff += abs((nn_score - theoretical_score(input, cov, mean)).sum())
        avg_pdf_diff += abs(output - math.log(theoretical_pdf(input, cov, mean)))
    avg_pdf_diff /= len(testing)
    avg_score_diff /= len(testing)
    logging.info(f'pdf diff: {avg_pdf_diff}')
    logging.info(f'score diff: {avg_score_diff}')

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

    mean = torch.Tensor([1, 1, 1])
    cov = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
    write_samples(distr)
    training, testing = get_samples()

    model = SM()
    optimiser = Adam(model.parameters(), lr = 0.75)

    train(model, optimiser, training, testing, cov, mean)
    torch.save(model.state_dict(), 'model/save.txt')
    

if __name__ == '__main__':
    main()