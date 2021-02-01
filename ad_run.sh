#!/bin/bash
EXP_NAME='20.02.01.exp1'
PATIENCE=5
STATE_DIM=22
HIDDEN_DIM=64
GRU_STEP=2
OPTIMIZER='Adam'
LR=0.001
MAX_EPOCHS=1

DATASET='cnsm_exp2_2' #'cnsm_exp1, cnsm_exp2_1, or cnsm_exp2_2'
DIRECTION='forward' #'bi-direction'
RECUR_W=0.7
REDUCE='mean' #'mean'

# check dataset and set csv paths
DATA_DIR='./data/'$DATASET'_data/gnn_data/'
if [ $DATASET = 'cnsm_exp1' ]
then
    CSV1='rnn_len16.fw.csv'
    CSV2='rnn_len16.ids.csv'
    CSV3='rnn_len16.flowmon.csv'
    CSV4='rnn_len16.dpi.csv'
    CSV5='rnn_len16.lb.csv'
    CSV_LABEL='rnn_len16.label.csv'
    
    N_NODES=5
else
    CSV1='rnn_len16.fw.csv'
    CSV2='rnn_len16.flowmon.csv'
    CSV3='rnn_len16.dpi.csv'
    CSV4='rnn_len16.ids.csv'
    CSV5=''
    CSV_LABEL='rnn_len16.label.csv'

    N_NODES=4
fi

export CUDA_VISIBLE_DEVICES=$1

for i in 1 2 3
do
    echo $i'th run'
    python3 ad_main.py  --exp_name=$EXP_NAME \
                        --patience=$PATIENCE \
                        --state_dim=$STATE_DIM \
                        --hidden_dim=$HIDDEN_DIM \
                        --GRU_step=$GRU_STEP \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --direction=$DIRECTION \
                        --reduce=$REDUCE \
                        --dataset=$DATASET \
                        --data_dir=$DATA_DIR \
                        --csv1=$CSV1 \
                        --csv2=$CSV2 \
                        --csv3=$CSV3 \
                        --csv4=$CSV4 \
                        --csv5=$CSV5 \
                        --csv_label=$CSV_LABEL \
                        --n_nodes=$N_NODES \
                        --recur_w=$RECUR_W \
                        --max_epochs=$MAX_EPOCHS
done
