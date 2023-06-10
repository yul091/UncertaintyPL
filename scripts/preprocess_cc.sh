PROJECT=different_author # different_author, different_project, different_time, case_study
DATA_DIR=data/main/$PROJECT
DEST_DIR=dataset_new/code_completion/$PROJECT

python -m program_tasks.code_completion.prepro \
    --data_dir $DATA_DIR \
    --dest_dir $DEST_DIR