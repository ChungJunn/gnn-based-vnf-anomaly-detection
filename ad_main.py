'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle as pkl

import math
import sys
import time

import argparse
import neptune

# declare model



# declare dataset


# modify the dataset to produce labels


# create a training loop


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', type=str, help='', default='')
    parser.add_argument('--val_path', type=str, help='', default='')
    parser.add_argument('--test_path', type=str, help='', default='')
    parser.add_argument('--stat_file', type=str, help='', default='')
    parser.add_argument('--batch_size', type=int, help='', default=0)
    parser.add_argument('--lr', type=float, help='', default=0.0)
    parser.add_argument('--optimizer', type=str, help='', default='SGD')
    parser.add_argument('--max_epoch', type=int, help='', default=1000)
    parser.add_argument('--valid_every', type=int, help='', default=1)
    parser.add_argument('--patience', type=int, help='', default=10)
    parser.add_argument('--dim_input', type=int, help='', default=88)
    parser.add_argument('--dim_out', type=int, help='', default=2)
    parser.add_argument('--drop_p', type=float, help='', default=0.3)
    parser.add_argument('--n_cmt', type=int, help='', default=1)
    parser.add_argument('--weight_decay', type=float, help='', default=0)
    parser.add_argument('--momentum', type=float, help='', default=0)
    parser.add_argument('--dim_hidden1', type=int, help='', default=0)
    parser.add_argument('--dim_hidden2', type=int, help='', default=0)
    parser.add_argument('--dim_hidden3', type=int, help='', default=0)

    parser.add_argument('--name', type=str, help='', default='')
    parser.add_argument('--tag', type=str, help='', default='')
    parser.add_argument('--out_file', type=str, help='', default='')
    args = parser.parse_args()
    params = vars(args)

    neptune.init('cjlee/AnomalyDetection-Supervised')
    experiment = neptune.create_experiment(name=args.name, params=params)
    args.out_file = experiment.id + '.pth'
    neptune.append_tag(args.tag)

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    # temporary code for testing
    train_main(args, neptune)
