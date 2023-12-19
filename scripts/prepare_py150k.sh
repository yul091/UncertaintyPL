# PYTHON150K_DIR=Python150kExtractor
# cd $PYTHON150K_DIR

# wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz

# mkdir py150_files
# tar -C py150_files -zxvf py150_files.tar.gz
# rm py150_files.tar.gz

# cd py150_files
# tar -zxvf data.tar.gz
# rm data.tar.gz

# cd ..

OUTPUT_DIR=data/preprocessed/py150
# Preprocess for training
# python preprocess.py --base_dir=py150_files --output_dir=$OUTPUT_DIR

DEST_DIR=dataset/code_completion/ood
python -B -m program_tasks.code_completion.prepro \
    --data_dir $OUTPUT_DIR \
    --dest_dir $DEST_DIR \
    --language python
