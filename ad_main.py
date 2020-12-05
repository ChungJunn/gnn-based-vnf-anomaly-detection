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

    for li, (anno, A_in, A_out, label, end_of_data) in enumerate(validiter):
        anno = anno.to(dtype=torch.float32, device=device)
        A_in = A_in.to(dtype=torch.float32, device=device)
        A_out = A_out.to(dtype=torch.float32, device=device)
        label = label.to(dtype=torch.int64, device=device)

        # go through loss function
        output = model(A_in, A_out, anno)
        loss = criterion(output, label)

        # compute loss
        valid_loss += loss.item()
        if end_of_data == 1: break

    valid_loss /= (li+1)

    return valid_loss

def train_main(args, neptune):
    device = torch.device('cuda:0')
    criterion = F.nll_loss

    # declare model
    model = gnn_binary_classifier(args).to(device)

    # declare dataset
    trainiter = ad_gnn_iterator(args, 'sup_train')
    valiter = ad_gnn_iterator(args, 'sup_val')
    testiter = ad_gnn_iterator(args, 'sup_test')

    # declare optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    train_loss2 = 0.0
    log_interval=1000
    bc = 0
    best_val = None
    savedir = './result/' + args.out_file

    for ei in range(1000):
        for li, (anno, A_in, A_out, label, end_of_data) in enumerate(trainiter):
            anno = anno.to(dtype=torch.float32, device=device)
            A_in = A_in.to(dtype=torch.float32, device=device)
            A_out = A_out.to(dtype=torch.float32, device=device)
            label = label.to(dtype=torch.int64, device=device)
        
            optimizer.zero_grad()

            output = model(A_in, A_out, anno)

            # go through loss function
            loss = criterion(output, label)
            loss.backward()

            # optimizer
            optimizer.step()
            train_loss += loss.item()
        
            if li % log_interval == (log_interval - 1):
                train_loss = train_loss / log_interval
                print('epoch: {:d} | li: {:d} | train_loss: {:.4f}'.format(ei+1, li+1, train_loss))
                neptune.log_metric('train loss', li, train_loss)
                train_loss = 0

            if end_of_data == 1: break

        # evaluation code
        valid_loss = validate(model, valiter, device, criterion)
        print('epoch: {:d} | li: {:d} | valid_loss: {:.4f}'.format(ei+1, li+1, valid_loss))
        neptune.log_metric('valid loss', ei, valid_loss)

        # need to implement early-stop
        if ei == 0 or valid_loss < best_val:
            torch.save(model, savedir)
            bc = 0
            best_val = valid_loss
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                print('early stopping..')
                break
            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    from ad_eval import eval_main
    eval_main(model, testiter, device, neptune=neptune)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='', default='exp_name')
    parser.add_argument('--patience', type=int, help='', default=5)
    parser.add_argument('--state_dim', type=int, help='', default=22)
    parser.add_argument('--hidden_dim', type=int, help='', default=64)
    parser.add_argument('--GRU_step', type=int, help='', default=5)
    parser.add_argument('--encoder_type', type=int, help='', default=2, choices=range(1,3))
    parser.add_argument('--lr', type=float, help='', default=0.001)
    parser.add_argument('--tvt', type=float, help='', default=0.001)
    args = parser.parse_args()

    params = vars(args)

    neptune.init('cjlee/AnomalyDetection-GNN')
    experiment = neptune.create_experiment(name=args.exp_name, params=params)
    args.out_file = experiment.id + '.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
