#!/bin/bash
EXP_NAME='exp'
PATIENCE=5
USE_EDGE=0
STATE_DIM=22 # 22 if not use edge
HIDDEN_DIM=64
GRU_STEP=5
OPTIMIZER='SGD'
LR=0.001
OUT_FILE='default.pth'
DIRECTION='forward'
RECUR_P=0.7

export CUDA_VISIBLE_DEVICES=$1

python3 ad_main.py  --exp_name=$EXP_NAME \
                    --patience=$PATIENCE \
                    --use_edge=$USE_EDGE \
                    --state_dim=$STATE_DIM \
                    --hidden_dim=$HIDDEN_DIM \
                    --GRU_step=$GRU_STEP \
                    --optimizer=$OPTIMIZER \
                    --lr=$LR \
                    --out_file=$OUT_FILE \
                    --direction=$DIRECTION \
