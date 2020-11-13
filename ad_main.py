'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import math
import sys
import time

import argparse
import neptune

from ad_model import gnn_binary_classifier
from ad_data import ad_gnn_iterator

def validate(model, validiter, device, criterion):
    valid_loss = 0.0

    for li, (anno, A_out, A_in, label, end_of_data) in enumerate(validiter):
        anno = anno.to(dtype=torch.float32, device=device)
        A_out = A_out.to(dtype=torch.float32, device=device)
        A_in = A_in.to(dtype=torch.float32, device=device)
        label = label.to(dtype=torch.int64, device=device)

        # go through loss function
        output = model(A_out, A_in, anno)
        loss = criterion(output, label)

        # compute loss
        valid_loss += loss.item()
        if end_of_data == 1: break

    valid_loss /= (li+1)

    return valid_loss

def train_main(args):
    device = torch.device('cuda:0')
    criterion = F.nll_loss

    # declare model
    model = gnn_binary_classifier(args).to(device)

    # declare dataset
    trainiter = ad_gnn_iterator()

    # declare optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0

    log_interval=1000

    for ei in range(1000):
        for li, (anno, A_out, A_in, label, end_of_data) in enumerate(trainiter):
            anno = anno.to(dtype=torch.float32, device=device)
            A_out = A_out.to(dtype=torch.float32, device=device)
            A_in = A_in.to(dtype=torch.float32, device=device)
            label = label.to(dtype=torch.int64, device=device)
        
            optimizer.zero_grad()

            output = model(A_out, A_in, anno)

            # go through loss function
            loss = criterion(output, label)
            loss.backward()

            # optimizer
            optimizer.step()
            train_loss += loss.item()
        
            if li % log_interval == 999:
                train_loss /= log_interval
                print('epoch: {:d} | li: {:d} | train_loss: {:.4f}'.format(ei+1, li+1, train_loss))
                train_loss = 0

            if end_of_data == 1: break

        # evaluation code
        valid_loss = validate(model, trainiter, device, criterion)
        print('epoch: {:d} | li: {:d} | valid_loss: {:.4f}'.format(ei+1, li+1, valid_loss))

        from ad_eval import eval_main
        eval_main(model, trainiter, device, neptune=None)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dim', type=int, help='', default=21)
    parser.add_argument('--hidden_dim', type=int, help='', default=64)
    parser.add_argument('--GRU_step', type=int, help='', default=5)
    parser.add_argument('--lr', type=float, help='', default=0.001)
    args = parser.parse_args()

    params = vars(args)

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    # temporary code for testing
    train_main(args)
