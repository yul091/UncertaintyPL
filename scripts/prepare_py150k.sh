PYTHON150K_DIR=Python150kExtractor
cd $PYTHON150K_DIR
wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
tar -xzvf py150.tar.gz

# Extract the data
mkdir -p $(pwd)/../data/py150
DATA_DIR=$(pwd)/../data/py150
SEED=239
python extract.py \
    --data_dir=$(pwd) \
    --output_dir=$DATA_DIR \
    --seed=$SEED

# Preprocess for training
# bash preprocess.sh $DATA_DIR