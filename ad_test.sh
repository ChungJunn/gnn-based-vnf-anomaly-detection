#!/bin/bash

EXP_NAME='20.12.24-testing'
MODEL_PATH='./result/AN3-240.pth'
DATASET='cnsm_exp1'
DIRECTION='forward'

export CUDA_VISIBLE_DEVICES=$1

python3 ad_test.py \
        --exp_name=$EXP_NAME \
        --model_path=$MODEL_PATH \
        --dataset=$DATASET \
        --direction=$DIRECTION
