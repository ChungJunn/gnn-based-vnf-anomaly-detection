#!/bin/bash
EXP_NAME='21.02.16.exp.debug'

OPTIMIZER='Adam'
LR=0.001
REDUCE='max' # max, mean

# GRU parameters
STATE_DIM=22
HIDDEN_DIM=64
GRU_STEP=5
DIRECTION='forward'
RECUR_W=0.5

# other fixed params
PATIENCE=20
MAX_EPOCH=1000

# check dataset and set csv paths
DATA_DIR=$HOME'/autoregressor/data/cnsm_exp1_data/gnn_data/'
DATA_DIR2=$HOME'/autoregressor/data/cnsm_exp2_2_data/gnn_data/'

CSV1='rnn_len16.fw.csv'
CSV2='rnn_len16.ids.csv'
CSV3='rnn_len16.flowmon.csv'
CSV4='rnn_len16.dpi.csv'
CSV5='rnn_len16.lb.csv'
CSV_LABEL='rnn_len16.label.csv'
    
N_NODES=5

ALPHA=0.5

export CUDA_VISIBLE_DEVICES=$1

for i in 1 2 3 4 5 
do
/usr/bin/python3.8 ad_dual_main.py  --data_dir=$DATA_DIR \
                    --data_dir2=$DATA_DIR2 \
                    --csv1=$CSV1 \
                    --csv2=$CSV2 \
                    --csv3=$CSV3 \
                    --csv4=$CSV4 \
                    --csv5=$CSV5 \
                    --csv_label=$CSV_LABEL \
                    --n_nodes=$N_NODES \
                    --reduce=$REDUCE \
                    --optimizer=$OPTIMIZER \
                    --lr=$LR \
                    --patience=$PATIENCE \
                    --exp_name=$EXP_NAME \
                    --max_epoch=$MAX_EPOCH \
                    --alpha=$ALPHA \
                    --state_dim=$STATE_DIM \
                    --hidden_dim=$HIDDEN_DIM \
                    --GRU_step=$GRU_STEP \
                    --direction=$DIRECTION \
                    --recur_w=$RECUR_W
done
