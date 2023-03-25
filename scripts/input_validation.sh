
SHIFT=different_time # different_project, different_time, different_author
TASK=code_summary # code_summary, code_completion
MODEL=code2vec # code2vec, coderoberta, graphcodebert, lstm, codebert, codegpt
DATADIR=dataset
MODELDIR=results
METRICDIR=Uncertainty_Results
OUTDIR=Uncertainty_Eval/filter

CUDA_VISIBLE_DEVICES=1 python filter.py \
    --shift_type $SHIFT \
    --task $TASK \
    --model $MODEL \
    --data_dir $DATADIR \
    --model_dir $MODELDIR \
    --uncertainty_dir $METRICDIR \
    --out_dir $OUTDIR