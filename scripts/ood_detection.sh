
TASK=code_completion
MODEL=codellama

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

for SHIFT in case_study; do
  RES_DIR=Uncertainty_logging/$TASK/$SHIFT
  EXPERIMENT_LOG=$RES_DIR/$MODEL'_miscls_prediction.txt'
  python -B -m Uncertainty_Eval.evaluation \
      --shift_type $SHIFT \
      --task $TASK \
      --ood \
      --model $MODEL | tee $EXPERIMENT_LOG
done