# make GNN model which use GG-NN based on GRU
import torch
import torch.nn as nn
import numpy as np
import torch.nn.funtional as F
from torch.autograd import Variable

class gnn_binary_classifier:
    def __init__(self, args):
        self.GRUcell = nn.GRUCell(2 * args.state_dim, args.state_dim, bias=False)
        self.fc1 = self.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = self.Linear(args.hidden_dim, 2)

    def encoder(self, A_out, A_in, annotation):
        N = self.n_nodes
        h = annotation

        for i in range(self.args.GRU_step):
            a_out = torch.matmul(A_out, h)
            a_in = torch.matmul(A_in, h)
            a = torch.cat((a_out, a_in), dim=1)
            h = self.GRUcell(a, h)

        enc_out = h

        return enc_out

    def classifier(self, enc_out):

        x = self.fc1(enc_out)
        x = self.fc2(enc_out)

        return F.log_softmax(x)

    def forward(self, A_out, A_in, annotation):

        x = self.encoder(A_out, A_in, annotation)
        x = self.classifier(x)

        return x

#### TODO ####
# trainloader and training


# setup args variables


# setup variables 

