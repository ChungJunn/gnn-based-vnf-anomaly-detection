#!/bin/bash
EXP_NAME='20.12.22.exp1'
PATIENCE=5
STATE_DIM=22
HIDDEN_DIM=64
GRU_STEP=5
OPTIMIZER='SGD'
LR=0.001
OUT_FILE='default.pth'
DIRECTION=$2 #'bi-direction'
RECUR_P=0.7
REDUCE=$3 #'mean'

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
                        --out_file=$OUT_FILE \
                        --direction=$DIRECTION \
                        --reduce=$REDUCE
done
