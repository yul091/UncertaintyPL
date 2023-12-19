PYTHON150K_DIR=Python150kExtractor
cd $PYTHON150K_DIR
# wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
# tar -xzvf py150.tar.gz

# # Extract the data
# mkdir -p $(pwd)/../data/py150
# DATA_DIR=$(pwd)/../data/py150
# SEED=239
# python extract.py \
#     --data_dir=$(pwd) \
#     --output_dir=$DATA_DIR \
#     --seed=$SEED

# BASE_DIR=$(pwd)/py150
# OUTPUT_DIR=$(pwd)/../data/preprocessed/py150
# python preprocess.py \
#     --base_dir $BASE_DIR \
#     --output_dir $OUTPUT_DIR


wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz

mkdir py150_files
tar -C py150_files -zxvf py150_files.tar.gz
rm py150_files.tar.gz

cd py150_files
tar -zxvf data.tar.gz
rm data.tar.gz

OUTPUT_DIR=$(pwd)/../dataset/code_completion/py150
# Preprocess for training
python preprocess.py --base_dir=py150_files --output_dir=$OUTPUT_DIR
