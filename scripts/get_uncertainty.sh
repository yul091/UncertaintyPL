#!/bin/bash

###################################################################################
# MODULE_ID=0 # 0 is code summary
# MODEL_TYPE=graphcodebert # code2vec, coderoberta, graphcodebert
# SHIFT_TYPE=case_study # different_time, different_project, different_author
# DATA_DIR=dataset/code_summary/$SHIFT_TYPE/java_pkl
# RES_DIR=results/code_summary/$SHIFT_TYPE/$MODEL_TYPE
# SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
# TRAIN_BATCH_SIZE=128
# TEST_BATCH_SIZE=128

# if [ ! -d $SAVE_DIR ]; then
#   mkdir -p $SAVE_DIR
# else
#   echo dir exist
# fi

# CUDA_VISIBLE_DEVICES=7 python test_uncertainty.py \
#   --module_id $MODULE_ID --res_dir $RES_DIR \
#   --data_dir $DATA_DIR --save_dir $SAVE_DIR \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --test_batch_size $TEST_BATCH_SIZE
###################################################################################

###################################################################################
MODULE_ID=1 # 1 is code completion
MODEL_TYPE=codegpt # lstm, codebert, codegpt
SHIFT_TYPE=case_study # different_time, different_project, different_author
DATA_DIR=dataset/code_completion/$SHIFT_TYPE/java_project/
RES_DIR=results/code_completion/$SHIFT_TYPE/$MODEL_TYPE
SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
MAX_SIZE=200

if [ ! -d $SAVE_DIR ]; then
  mkdir -p $SAVE_DIR
else
  echo dir exist
fi

CUDA_VISIBLE_DEVICES=4 python test_uncertainty.py \
  --module_id $MODULE_ID --res_dir $RES_DIR \
  --data_dir $DATA_DIR --save_dir $SAVE_DIR \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --test_batch_size $TEST_BATCH_SIZE \
  --max_size $MAX_SIZE
###################################################################################


