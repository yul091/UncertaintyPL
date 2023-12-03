
TASK=code_completion
SHIFT=different_author
RES_DIR=Uncertainty_logging/$TASK/$SHIFT
MODEL=codellama

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

for SHIFT in different_author different_time different_project case_study; do
  EXPERIMENT_LOG=$RES_DIR/$MODEL'.txt'
  python Uncertainty_Eval/evaluation.py \
      --shift_type $SHIFT \
      --task $TASK \
      --model $MODEL | tee $EXPERIMENT_LOG
done