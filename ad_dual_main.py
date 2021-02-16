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

from ad_eval import eval_main
from ad_data import ad_gnn_iterator
from ad_model import gnn_binary_classifier

def validate(model, validiter, device, criterion):
    valid_loss = 0.0

    for li, (anno, label, end_of_data) in enumerate(validiter):
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
    print('# model', model)

    # declare two datasets
    csv_files=[]
    for n in range(1, args.n_nodes+1):
        csv_file = eval('args.csv' + str(n))
        csv_files.append(csv_file)
    csv_files.append(args.csv_label) # append label 

    # declare dataset
    trainiter = ad_gnn_iterator(tvt='sup_train', data_dir=args.data_dir, csv_files=csv_files, direction=args.direction, recur_w=args.recur_w)
    valiter = ad_gnn_iterator(tvt='sup_val', data_dir=args.data_dir, csv_files=csv_files, direction=args.direction, recur_w=args.recur_w)
    testiter = ad_gnn_iterator(tvt='sup_test', data_dir=args.data_dir, csv_files=csv_files, direction=args.direction, recur_w=args.recur_w)

    # dual training
    ns = [1,3,4,2]
    csv_files2=[]
    for n in ns:
        csv_file = eval('args.csv' + str(n))
        csv_files2.append(csv_file)
    csv_files2.append(args.csv_label) # append label 

    trainiter2 = ad_gnn_iterator(tvt='sup_train', data_dir=args.data_dir2, csv_files=csv_files2, direction=args.direction, recur_w=args.recur_w)
    valiter2 = ad_gnn_iterator(tvt='sup_val', data_dir=args.data_dir2, csv_files=csv_files2, direction=args.direction, recur_w=args.recur_w)
    testiter2 = ad_gnn_iterator(tvt='sup_test', data_dir=args.data_dir2, csv_files=csv_files2, direction=args.direction, recur_w=args.recur_w)

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    log_interval=1000
    bc = 0
    best_val_f1 = None
    savedir = './result/' + args.out_file
    n_samples = trainiter.n_samples

    for ei in range(args.max_epoch):
        for li, (iter1_data, iter2_data) in enumerate(zip(trainiter, trainiter2)):
            anno, A_in, A_out, label, _ = iter1_data # don't use end_of_data from iter1 bc it has fewer number of data
            anno = anno.to(dtype=torch.float32, device=device)
            A_in = A_in.to(dtype=torch.float32, device=device)
            A_out = A_out.to(dtype=torch.float32, device=device)
            label = label.to(dtype=torch.int64, device=device)

            anno2, A_in2, A_out2, label2, end_of_data = iter2_data
            anno2 = anno2.to(dtype=torch.float32, device=device)
            A_in2 = A_in2.to(dtype=torch.float32, device=device)
            A_out2 = A_out2.to(dtype=torch.float32, device=device)
            label2 = label2.to(dtype=torch.int64, device=device)
        
            optimizer.zero_grad()

            output = model(A_in, A_out, anno)
            output2 = model(A_in2, A_out2, anno2)

            # go through loss function
            loss = criterion(output, label)
            loss2 = criterion(output2, label2)

            combined_loss = args.alpha * loss + (1-args.alpha) * loss2
            combined_loss.backward()

            # optimizer
            optimizer.step()
            train_loss += combined_loss.item()
        
            if li % log_interval == (log_interval - 1):
                train_loss = train_loss / log_interval
                print('epoch: {:d} | li: {:d} | train_loss: {:.4f}'.format(ei+1, li+1, train_loss))
                if neptune is not None: neptune.log_metric('train loss', (ei*n_samples)+(li+1), train_loss)
                train_loss = 0

            if end_of_data == 1: break

        # evaluation code
        # valid_loss = validate(model, valiter, device, criterion)
        acc,prec,rec,f1=eval_main(model,valiter,device,neptune=None)
        acc2,prec2,rec2,f12=eval_main(model,valiter2,device,neptune=None)
        combined_f1= args.alpha * f1 + (1-args.alpha) * f12
        print('epoch: {:d} | valid_f1: {:.4f}'.format(ei+1, combined_f1))
        if neptune is not None: neptune.log_metric('valid f1', ei, combined_f1)

        # need to implement early-stop
        if ei == 0 or combined_f1 > best_val_f1:
            torch.save(model, savedir)
            bc = 0
            best_val_f1=combined_f1
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                print('early stopping..')
                break
            print('bad counter == %d' % (bc))

    model = torch.load(savedir)
    acc, prec, rec, f1 = eval_main(model, testiter, device, neptune=neptune)
    acc2, prec2, rec2, f12 = eval_main(model, testiter2, device, neptune=neptune)

    if neptune is not None:
        neptune.set_property('cnsm_exp1_acc', acc)
        neptune.set_property('cnsm_exp1_prec', prec)
        neptune.set_property('cnsm_exp1_rec', rec)
        neptune.set_property('cnsm_exp1_f1', f1)

        neptune.set_property('cnsm_exp2_2_acc', acc2)
        neptune.set_property('cnsm_exp2_2_prec', prec2)
        neptune.set_property('cnsm_exp2_2_rec', rec2)
        neptune.set_property('cnsm_exp2_2_f1', f12)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--csv1', type=str)
    parser.add_argument('--csv2', type=str)
    parser.add_argument('--csv3', type=str)
    parser.add_argument('--csv4', type=str)
    parser.add_argument('--csv5', type=str)
    parser.add_argument('--csv_label', type=str)

    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--reduce', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--max_epoch', type=int)

    # dual training
    parser.add_argument('--data_dir2', type=str)
    parser.add_argument('--alpha', type=float)

    # for GNN encoder + classifier
    parser.add_argument('--GRU_step', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--state_dim', type=int)
    parser.add_argument('--direction', type=str)
    parser.add_argument('--recur_w', type=float)

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
