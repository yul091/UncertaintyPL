
SHIFT=case_study # different_project, different_time, different_author
TASK=code_completion # code_summary, code_completion
MODEL=codegpt # code2vec, coderoberta, graphcodebert, lstm, codebert, codegpt

DATADIR=dataset
MODELDIR=results
METRICDIR=Uncertainty_Results
OUTDIR=Uncertainty_Eval/input_validation

CUDA_VISIBLE_DEVICES=7 python filter.py \
    --shift_type $SHIFT \
    --task $TASK \
    --model $MODEL \
    --data_dir $DATADIR \
    --model_dir $MODELDIR \
    --uncertainty_dir $METRICDIR \
    --out_dir $OUTDIR \
    --strategy coverage