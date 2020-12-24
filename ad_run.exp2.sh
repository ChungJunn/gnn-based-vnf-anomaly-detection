#!/bin/bash

DATA1='cnsm_exp1'
DATA2='cnsm_exp2_1'
DATA3='cnsm_exp2_2'

./ad_run.sh $1 $DATA1 $3 $4 $5
./ad_run.sh $1 $DATA2 $3 $4 $5
./ad_run.sh $1 $DATA3 $3 $4 $5
