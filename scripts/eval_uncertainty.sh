
TASK=code_summary
SHIFT=different_author
RES_DIR=Uncertainty_logging/$TASK/$SHIFT

if [ ! -d $RES_DIR ]; then
  mkdir -p $RES_DIR
else
  echo dir exist
fi

for MODEL in code2vec coderoberta graphcodebert lstm codebert codegpt
do
    EXPERIMENT_LOG=$RES_DIR/$MODEL'.txt'
    python Uncertainty_Eval/evaluation.py \
        --shift_type $SHIFT \
        --task $TASK \
        --model $MODEL | tee $EXPERIMENT_LOG
done

# for MODEL in code2vec coderoberta graphcodebert
# do
#     for SHIFT in different_time different_project different_author
#     do
#         python Uncertainty_Eval/evaluation.py \
#         --shift_type $SHIFT \
#         --task $TASK \
#         --model $MODEL | tee $EXPERIMENT_LOG
#     done
# done


# TASK=code_completion
# for MODEL in lstm codebert codegpt
# do
#     for SHIFT in different_time different_project different_author
#     do
#         python Uncertainty_Eval/evaluation.py \
#         --shift_type $SHIFT \
#         --task $TASK \
#         --model $MODEL | tee $EXPERIMENT_LOG
#     done
# done


# SHIFT=case_study
# TASK=code_summary
# for MODEL in code2vec coderoberta graphcodebert
# do
#     python Uncertainty_Eval/evaluation.py \
#     --shift_type $SHIFT \
#     --task $TASK \
#     --model $MODEL
# done


# TASK=code_completion
# for MODEL in lstm codebert codegpt
# do
#     python Uncertainty_Eval/evaluation.py \
#     --shift_type $SHIFT \
#     --task $TASK \
#     --model $MODEL
# done