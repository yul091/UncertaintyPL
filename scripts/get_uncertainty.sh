#!/bin/bash


###################################################################################
MODULE_ID=0 # 0 is code summary
MODEL_TYPE=codegpt # code2vec, codeberta, graphcodebert, lstm, codebert, codegpt
SHIFT_TYPE=different_time # different_time, different_project, different_author
DATA_DIR=dataset/code_summary/$SHIFT_TYPE/java_pkl/
RES_DIR=results_new/code_summary/$SHIFT_TYPE/$MODEL_TYPE
SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
TRAIN_BATCH_SIZE=8
TEST_BATCH_SIZE=8
LOG_DIR=Uncertainty_logging/code_summary/$SHIFT_TYPE
EXPERIMENT_LOG=$LOG_DIR/$MODEL_TYPE'.txt'

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
else
  echo dir exist
fi

if [ ! -d $SAVE_DIR ]; then
  mkdir -p $SAVE_DIR
else
  echo dir exist
fi

CUDA_VISIBLE_DEVICES=5 python test_uncertainty.py \
  --module_id $MODULE_ID --res_dir $RES_DIR \
  --data_dir $DATA_DIR --save_dir $SAVE_DIR \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --test_batch_size $TEST_BATCH_SIZE | tee $EXPERIMENT_LOG
###################################################################################

###################################################################################
# MODULE_ID=1 # 1 is code completion
# MODEL_TYPE=graphcodebert # lstm, codebert, codegpt, code2vec, codeberta, graphcodebert
# SHIFT_TYPE=different_time # different_time, different_project, different_author
# DATA_DIR=dataset/code_completion/$SHIFT_TYPE/java_project/
# RES_DIR=results_new/code_completion/$SHIFT_TYPE/$MODEL_TYPE
# SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
# TRAIN_BATCH_SIZE=8
# TEST_BATCH_SIZE=8
# MAX_SIZE=200
# LOG_DIR=Uncertainty_logging/code_completion/$SHIFT_TYPE
# EXPERIMENT_LOG=$LOG_DIR/$MODEL_TYPE'.txt'

# if [ ! -d $LOG_DIR ]; then
#   mkdir -p $LOG_DIR
# else
#   echo dir exist
# fi

# if [ ! -d $SAVE_DIR ]; then
#   mkdir -p $SAVE_DIR
# else
#   echo dir exist
# fi

# CUDA_VISIBLE_DEVICES=2 python test_uncertainty.py \
#   --module_id $MODULE_ID --res_dir $RES_DIR \
#   --data_dir $DATA_DIR --save_dir $SAVE_DIR \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --test_batch_size $TEST_BATCH_SIZE \
#   --max_size $MAX_SIZE | tee $EXPERIMENT_LOG
###################################################################################


