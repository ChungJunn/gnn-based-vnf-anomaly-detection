# make GNN model which use GG-NN based on GRU
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class gnn_binary_classifier(nn.Module):
    def __init__(self, args):
        super(gnn_binary_classifier, self).__init__()
        self.args = args
        self.n_nodes = 4

        self.GRUcell = nn.GRUCell(args.state_dim, args.state_dim, bias=False)
        self.fc1 = nn.Linear(args.state_dim * self.n_nodes, args.hidden_dim) # assume 4 nodes
        self.fc2 = nn.Linear(args.hidden_dim, 2)

    def encoder(self, A, annotation):
        h = annotation

        for i in range(self.args.GRU_step):
            a = torch.matmul(A, h)
            h = self.GRUcell(a, h)

        enc_out = h
        enc_out = enc_out.view(1,-1) # squash into single vector

        return enc_out

    def classifier(self, enc_out):

        x = self.fc1(enc_out)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def forward(self, A, annotation):

        x = self.encoder(A, annotation)
        x = self.classifier(x)

        return x

#### TODO ####
# trainloader and training

# setup args variables

# setup variables 

