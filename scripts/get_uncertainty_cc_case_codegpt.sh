#!/bin/bash

###################################################################################
MODULE_ID=1 # 1 is code completion
SHIFT_TYPE=case_study # different_time, different_project, different_author
DATA_DIR=dataset/code_completion/$SHIFT_TYPE
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=128
MAX_SIZE=200
LOG_DIR=Uncertainty_logging/code_completion/$SHIFT_TYPE

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
else
  echo dir exist
fi

for MODEL_TYPE in codegpt; do
  RES_DIR=results/code_completion/$SHIFT_TYPE/$MODEL_TYPE
  SAVE_DIR=Uncertainty_Results/$SHIFT_TYPE/$MODEL_TYPE
  EXPERIMENT_LOG=$LOG_DIR/$MODEL_TYPE'.txt'
  for dir in "${RES_DIR}"/ensemble_model-*; do
      # Check if it's indeed a directory
      # if [ -d "$dir" ]; then
      if [ -d "$dir" ] && [ "$(basename "$dir")" != "ensemble_model-0" ]; then
          # Append to the ENSEMBLE_DIRS string, separated by space
          ENSEMBLE_DIRS="${ENSEMBLE_DIRS} ${dir}"
      fi
  done

  if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
  else
    echo dir exist
  fi

  CUDA_VISIBLE_DEVICES=3 python test_uncertainty.py \
    --module_id $MODULE_ID --res_dir $RES_DIR/ensemble_model-0 \
    --data_dir $DATA_DIR --save_dir $SAVE_DIR \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --ensemble_dirs $ENSEMBLE_DIRS \
    --max_size $MAX_SIZE | tee $EXPERIMENT_LOG
done
###################################################################################


