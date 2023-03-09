# UncertaintyPL
An Impact Study of Data Shift Towards Code Analysis Model and Uncertainty Measurement:

This is a PyTorch implementation of two code analysis tasks, namely, code summarization and code completion with three model archiectures for each task. We use the Java and Python extractors to preprocess the raw code snippets. Users may further extend the work to other programming languages following our study.

<p align="center">
  <img src="Figure/code_shift.png" width="70%" height="70%">
</p>

Our study overview:
<p align="center">
  <img src="Figure/study_overview.png" width="100%" height="100%">
</p>

Code analysis mechanism:
<p align="center">
  <img src="Figure/code_analysis.png" width="100%" height="100%">
</p>

## Requirements
- [python3.8+](https://www.python.org/downloads/release/python-380/)
- [PyTorch 1.13.0](https://pytorch.org/get-started/locally/)
- Libraries and dependencies:
```
pip install -r requirements.txt
```

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
- Edit the file [scripts/train_cs.sh](scripts/train_cs.sh) and file [scripts/train_cc.sh](scripts/train_cc.sh) to point to the right preprocessed data and a specific model archiecture.
- Before training, you can edit the configuration hyper-parameters in these two files.
- Run the two shell scripts:
```
bash scripts/train_cs.sh # code summary
bash scripts/train_cc.sh # code completion
```
### Step 3: Measuring the five uncertainty scores
- Edit the file [scripts/get_uncertainty.sh](scripts/get_uncertainty.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [scripts/get_uncertainty.sh](scripts/get_uncertainty.sh):
```
bash scripts/get_uncertainty.sh
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