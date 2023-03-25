
# TASK=code_summary
# for MODEL in code2vec coderoberta graphcodebert
# do
#     for SHIFT in different_time different_project different_author
#     do
#         python Uncertainty_Eval/evaluation.py \
#         --shift_type $SHIFT \
#         --task $TASK \
#         --model $MODEL
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
#         --model $MODEL
#     done
# done


SHIFT=case_study
TASK=code_summary
for MODEL in code2vec coderoberta graphcodebert
do
    python Uncertainty_Eval/evaluation.py \
    --shift_type $SHIFT \
    --task $TASK \
    --model $MODEL
done


TASK=code_completion
for MODEL in lstm codebert codegpt
do
    python Uncertainty_Eval/evaluation.py \
    --shift_type $SHIFT \
    --task $TASK \
    --model $MODEL
done