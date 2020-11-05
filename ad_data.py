import pandas as pd
import numpy as np
import torch

# create annotation and adjacency matrices and dataloader
class ad_gnn_iterator:
    def __init__(self):
    # obtain dataset
        fw_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.fw.csv'
        flowmon_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.flowmon.csv'
        dpi_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.dpi.csv'
        ids_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.ids.csv'
        edge_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.edges.csv'
        label_path = '/home/mi-lab02/autoregressor/data/cnsm_exp2_2_data/gnn_data/sup_train.rnn_len16.label.csv'

        self.firewall= np.array(pd.read_csv(fw_path))
        self.flowmon= np.array(pd.read_csv(flowmon_path))
        self.dpi= np.array(pd.read_csv(dpi_path))
        self.ids= np.array(pd.read_csv(ids_path))

        self.edges = np.array(pd.read_csv(edge_path))
        self.label = np.array(pd.read_csv(label_path))

        # initialize some stuff
        self.node_features = [self.firewall,
                              self.flowmon,
                              self.dpi,
                              self.ids]
        self.idx = 0
        self.n_samples = self.firewall.shape[0]

        # initialize the variables
        self.n_nodes = 4
        self.n_node_features = self.firewall.shape[1]
        self.n_edge_features = 1

    def make_annotation_matrix(self, idx):
        # initialize the matrix
        annotation = np.zeros([self.n_nodes, self.n_node_features])

        # retrieve the related data using idx
        for ni in range(self.n_nodes):
            for fi in range(self.n_node_features):
                annotation[ni,fi] = (self.node_features[ni])[idx, fi]

        return annotation

    def make_adj_matrix(self, idx):
        # initialize the matrix
        A_in = np.zeros([self.n_nodes, self.n_nodes])
        A_out = np.zeros([self.n_nodes, self.n_nodes])

        # retrieve the related data using idx
        for from_node in range(self.n_nodes - 1): # no edge feature for last node
            A_in[from_node, from_node + 1] = self.edges[idx, from_node]

        A_out = A_in.transpose()

        # return the adj_matrix
        return A_out, A_in

    def __reset__(self):
        self.idx = 0
        return

    def __next__(self):
        if self.idx >= self.n_samples:
            raise StopIteration
            self.reset()

        annotation = self.make_annotation_matrix(self.idx)
        A_out, A_in = self.make_adj_matrix(self.idx)
        label = self.label[self.idx]

        self.idx += 1
        
        annotation = torch.tensor(annotation)
        A_out = torch.tensor(A_out)
        A_in = torch.tensor(A_in)
        label = torch.tensor(label)

        return annotation, A_out, A_in, label

    def __iter__(self):
        return self

if __name__ == '__main__':
    iter = ad_gnn_iterator()

    for iloop, (anno, A_out, A_in, label) in enumerate(iter):
        print(iloop, anno.shape, A_out.shape, A_in.shape, label.shape)
        print(label)
        
