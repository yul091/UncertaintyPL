# UncertaintyPL

## Requirements
- [python3](https://www.python.org/downloads/release/python-380/)
- [PyTorch 1.9.0](https://pytorch.org/get-started/locally/)

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/yul091/UncertaintyPL.git
cd UncertaintyPL
```
### Step 1: Download the preprocessed Java-small dataset (~60 K examples, compressed: 84MB) and Python150k dataset for OOD detection (~150 K examples, compressed: 526MB)
```
wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz
tar -xvzf java-small.tar.gz
wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
tar -xzvf py150.tar.gz
```
### Step 2: Training a model
#### Training a model from scratch
To train a model from scratch:
- Edit the file [program_tasks/code_summary/train.sh](program_tasks/code_summary/train.sh) and file [program_tasks/code_completion/train.sh](program_tasks/code_completion/train.sh) to point to the right preprocessed data and a specific model archiecture.
- Before training, you can edit the configuration hyper-parameters in these two files.
- Run the two shell scripts:
```
# code summary
bash program_tasks/code_summary/train.sh
# code completion
bash program_tasks/code_completion/train.sh
```
### Step 3: Measuring the five uncertainty scores
- Edit the file [Uncertainty_Eval/get_uncertainty.sh](Uncertainty_Eval/get_uncertainty.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [Uncertainty_Eval/get_uncertainty.sh](Uncertainty_Eval/get_uncertainty.sh):
```
bash Uncertainty_Study/get_uncertainty.sh
```
### Step 4: Evaluation the effectiveness of the five uncertainty methods on both error/success prediction and in-/out-of-distribution detection:
- Edit the file [Uncertainty_Eval/evaluation.py](Uncertainty_Eval/evaluation.py) to point to the target evaluation choice (error/success prediction or in-/out-of-distribution detection).
- Run the script [Uncertainty_Eval/evaluation.py](Uncertainty_Eval/evaluation.py):
```
python Uncertainty_Eval/evaluation.py
```
### Step 5: Evaluation the effectiveness of the five uncertainty methods on input validation:
- Edit the file [filter.py](filter.py) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [filter.py](filter.py):
```
python filter.py
```