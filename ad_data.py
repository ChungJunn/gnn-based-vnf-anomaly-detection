import pandas as pd
import numpy as np
import torch

# create annotation and adjacency matrices and dataloader
class ad_gnn_iterator:
    def __init__(self, tvt, dataset, direction='forward', recur_p=0.7):

        datasets = ['cnsm_exp1','cnsm_exp2_1','cnsm_exp2_2']
        if dataset not in datasets:
            print('dataset must be either cnsm_ex1, exp2_2, or exp2_2')
            import sys; sys.exit(-1)

        fw_path = '/home/mi-lab02/autoregressor/data/' + dataset +'_data/gnn_data/' + \
                tvt + '.rnn_len16.fw.csv'
        flowmon_path = '/home/mi-lab02/autoregressor/data/' + dataset + '_data/gnn_data/' + \
                tvt + '.rnn_len16.flowmon.csv'
        dpi_path = '/home/mi-lab02/autoregressor/data/' + dataset + '_data/gnn_data/' + \
                tvt + '.rnn_len16.dpi.csv'
        ids_path = '/home/mi-lab02/autoregressor/data/' + dataset + '_data/gnn_data/' + \
                tvt + '.rnn_len16.ids.csv'
        label_path = '/home/mi-lab02/autoregressor/data/' + dataset + '_data/gnn_data/' + \
                tvt + '.rnn_len16.label.csv'

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler()
        mm_scaler = MinMaxScaler()

        self.firewall= scaler.fit_transform(np.array(pd.read_csv(fw_path)))
        self.flowmon= scaler.fit_transform(np.array(pd.read_csv(flowmon_path)))
        self.dpi= scaler.fit_transform(np.array(pd.read_csv(dpi_path)))
        self.ids= scaler.fit_transform(np.array(pd.read_csv(ids_path)))

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

        self.direction = direction
        self.recur_p = recur_p

    def make_annotation_matrix(self, idx):
        # initialize the matrix
        annotation = np.zeros([self.n_nodes, self.n_node_features])

        # retrieve the related data using idx
        for ni in range(self.n_nodes):
            for fi in range(self.n_node_features):
                annotation[ni,fi] = (self.node_features[ni])[idx, fi]

        return annotation

    def make_adj_matrix(self, idx, direction='forward', recur_p=0.7):
        # initialize the matrix
        A_in = np.zeros([self.n_nodes, self.n_nodes])
        A_out = np.zeros([self.n_nodes, self.n_nodes])
        A_in += -np.inf

        n_edges = 1.0 if direction=='forward' else 2.0
        edge_weight = (1 - recur_p) / n_edges

        import math # retrieve the related data using idx
        for from_node in range(self.n_nodes): # no edge feature for last node
            A_in[from_node, from_node] = recur_p

            if from_node < (self.n_nodes - 1):
                A_in[from_node, from_node + 1] = edge_weight

            if direction == 'bi-direction' and from_node > 0:
                A_in[from_node, from_node - 1] = edge_weight

        # normalize using softmax
        from scipy.special import softmax
        A_in = softmax(A_in, axis=0)
        A_out += A_in

        return A_in, A_out

    def reset(self):
        self.idx = 0
        return

    def __next__(self):
        end_of_data=0

        if self.idx >= (self.n_samples - 1):
            end_of_data=1
            self.reset()

        annotation = self.make_annotation_matrix(self.idx)
        A_in, A_out = self.make_adj_matrix(self.idx, direction=self.direction,recur_p=self.recur_p)

        label = self.label[self.idx]

        self.idx += 1

        annotation = torch.tensor(annotation)

        A_in = torch.tensor(A_in)
        A_out = torch.tensor(A_out)
        label = torch.tensor(label)

        return annotation, A_in, A_out, label, end_of_data

    def __iter__(self):
        return self

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dim', type=int, help='', default=21)
    parser.add_argument('--hidden_dim', type=int, help='', default=64)
    parser.add_argument('--GRU_step', type=int, help='', default=5)
    parser.add_argument('--lr', type=float, help='', default=0.001)
    parser.add_argument('--tvt', type=str, help='', default='sup_val')
    parser.add_argument('--dataset', type=str, help='', default='cnsm_exp2_2')
    parser.add_argument('--direction', type=str, help='', default='forward')
    args = parser.parse_args()

    iter = ad_gnn_iterator(tvt = args.tvt, dataset=args.dataset, direction=args.direction)

    for iloop, (anno, A_out, A_in, label, end_of_data) in enumerate(iter):
        print(iloop, anno.shape, A_out.shape, A_in.shape, label.shape)
        print(label)
        import pdb; pdb.set_trace()
