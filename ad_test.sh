#!/bin/bash

EXP_NAME='20.12.22-testing'
MODEL_PATH='./result/AN3-154.pth'
DATASET='cnsm_exp2_2'
DIRECTION='bi-direction'

export CUDA_VISIBLE_DEVICES=$1

python3 ad_test.py \
        --exp_name=$EXP_NAME \
        --model_path=$MODEL_PATH \
        --dataset=$DATASET \
        --direction=$DIRECTION
