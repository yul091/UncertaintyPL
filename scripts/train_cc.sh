#!/bin/bash

MODEL_TYPE=codegpt # codebert, codegpt, lstm
SHIFT_TYPE=case_study # different_project, different_author, different_time
RES_DIR=results/code_completion/$SHIFT_TYPE/$MODEL_TYPE

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

DATA_DIR=dataset/code_completion/$SHIFT_TYPE/java_project
EPOCHS=100
BATCH=1024
LR=0.001
TRAIN_DATA=$DATA_DIR/train.tsv
VAL_DATA=$DATA_DIR/val.tsv
TEST_DATA=$DATA_DIR/test.tsv
# TEST_DATA1=$DATA_DIR/test1.tsv
# TEST_DATA2=$DATA_DIR/test2.tsv
# TEST_DATA3=$DATA_DIR/test3.tsv

EMBEDDING_TYPE=1
EMBEDDING_DIM=120 # dimension of vectors
EMBEDDING_PATH=/ # file for pre-trained vectors
EXPERIMENT_NAME=code_completion
EXPERIMENT_LOG=$RES_DIR/$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
# --test_data1 $TEST_DATA1 --test_data2 $TEST_DATA2 --test_data3 $TEST_DATA3 \

CUDA_VISIBLE_DEVICES=1 python -m program_tasks.code_completion.main \
  --train_data $TRAIN_DATA --val_data $VAL_DATA \
  --test_data $TEST_DATA \
  --model_type $MODEL_TYPE \
  --embedding_path $EMBEDDING_PATH \
  --embedding_type $EMBEDDING_TYPE \
  --embedding_dim $EMBEDDING_DIM \
  --epochs $EPOCHS --batch $BATCH --lr $LR --res_dir $RES_DIR \
  --experiment_name $EXPERIMENT_NAME | tee $EXPERIMENT_LOG

