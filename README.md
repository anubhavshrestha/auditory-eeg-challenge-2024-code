Auditory-eeg-challenge-2024-code
================================
This github repository documents my participation to the [2024 ICASSP Auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2024), conducted under the mentorship of Professor Tuka Alhanai with Human-Computer Interaction Lab at NYU Abu Dhabi.This codebase contains baseline models and code to preprocess stimuli for the tasks.

## Task1: Match-mismatch

The goal of this task was to determine which segment of speech matches the EEG given five segments of speech and a segment of EEG data. For this task, I implemented various Machine Learning models like CNN+RNN, LSTM, dilated LSTM, and so on. After trying various such models, I got upto 62% accuracy for the dilated CNN and LSTM model on the validation set.

## Task2: The second task was to reconstruct the mel spectorgram from the EEG. 

While I could not make significant contribution to this task, I have included general dataset, code for preprocessing the EEG and for creating commonly used stimulus representations, and two baseline methods as given by the ICASSP team.

# Prerequisites

Python >= 3.6

# General setup

Steps to get a working setup:

## 1. Clone this repository and install the [requirements.txt](requirements.txt)
```bash
# Clone this repository
git clone https://github.com/exporl/auditory-eeg-challenge-2024-code

# Go to the root folder
cd auditory-eeg-challenge-2024-code

# Optional: install a virtual environment
python3 -m venv venv # Optional
source venv/bin/activate # Optional

# Install requirements.txt
python3 -m install requirements.txt
```

## 2. [Download the data]

Download the dataset provided by the organizing team of this challenge. After downloading the dataset, the following information might help.

   1. `split_data(.zip)` contains already preprocessed, split and normalized data; ready for model training/evaluation. 
If you want to get started quickly, you can opt to only download this folder/zipfile.

   2. `preprocessed_eeg(.zip)` and `preprocessed_stimuli(.zip)` contain preprocessed EEG and stimuli files (envelope and mel features) respectively.
At this stage data is not yet split into different sets and normalized. To go from this to the data in `split_data`, you will have to run the `split_and_normalize.py` script ([preprocessing_code/split_and_normalize.py](./preprocessing_code/split_and_normalize.py) )

   3. `sub_*(.zip)` and `stimuli(.zip)` contain the raw EEG and stimuli files. 
If you want to recreate the preprocessing steps, you will need to download these files and then run `sparrKULee.py` [(preprocessing_code/sparrKULee.py)](./preprocessing_code/sparrKULee.py) to preprocess the EEG and stimuli and then run the `split_and_normalize.py` script to split and normalize the data.
It is possible to adapt the preprocessing steps in `sparrKULee.py` to your own needs, by adding/removing preprocessing steps. For more detailed information on the pipeline, see the [brain_pipe documentation](https://exporl.github.io/brain_pipe/).


## 3. Adjust the `config.json` accordingly

There is a general `config.json` defining the folder names and structure for the data (i.e. [util/config.json](./util/config.json) ).
Adjust `dataset_folder` in the `config.json` file from `null` to the absolute path to the folder containing all data (The `challenge_folder` from the previous point). 
If you follow the BIDS structure, by downloading the whole dataset, the folders preprocessed_eeg, preprocessed_stimuli and split_data, should be located inside the 'derivatives' folder. If you only download these three folders, make sure they are either in a subfolder 'derivatives', or change the 'derivatives' folder in the config, otherwise you will get a file-not-found error when trying to run the experiments. 
  

OK, you should be all setup now!

    

# Running the tasks

Each task has already some ready-to-go experiments files defined to give you a
baseline and make you acquainted with the problem. The experiment files live
in the `experiment` subfolder for each task. The training log,
best model and evaluation results will be stored in a folder called
`results_{experiment_name}`.

## Task1: Match-mismatch
    
By running task1_match_mismatch/experiments/model.py, you can run the relevant model for Task 1 challenges.

## Task2: Regression (reconstructing spectrogram from EEG)

By running [task2_regression/experiments/linear_baseline.py](./task2_regression/experiments/linear_baseline.py), you can 
train and evaluate a simple linear baseline model with Pearson correlation as a loss function, similar to the baseline model used in [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945).

By running [task2_regression/experiments/vlaai.py](./task2_regression/experiments/vlaai.py), you can train/evaluate
the VLAAI model as proposed by [Accou et al (2022)](https://www.biorxiv.org/content/10.1101/2022.09.28.509945). You can find a pre-trained model at [VLAAI's github page](https://github.com/exporl/vlaai).

Other models you might find interesting are: [Thornton et al. (2022)](https://iopscience.iop.org/article/10.1088/1741-2552/ac7976),...
