# kaggle: otto-Multi-Objective Recommender System
 My solution, which was in third place at the end of the competition, is now available to the public.

## Solution Overview
Makotu part of this URL for solutions.  
https://www.kaggle.com/competitions/otto-recommender-system/discussion/382879

## Code
Execute .ipynb and .py in order from 0 to 5 in the code folder.

Estimated execution times are as follows.  
preprocessing: about 1 day (especially clustering takes time)  
Make features: about 1 day  
Training & predict: about 1 day  

## Data
The data for the main source refers to data from radek.  
https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint  
https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation  
First put these data into "input/train_test" and "input/train_valid" before executing.

## Model
https://www.kaggle.com/datasets/mhyodo/otto-makotu-models

## for prediction
To reproduce the inference, take the following steps
1. store the original data in the input folder train_test and train_valid (please read the readme for those folders)
2. run code folder from 0-3 (up to 3_features)
3. store the above trained models in each model in the model folder
4. run prediction.py in 5 of the code folder

## for train
1. store the original data in the input folder train_test and train_valid (please read the readme for those folders)
2. run code folder from 0-3 (up to 3_features)
3. run train.py in 4 of the code folder

## Environment
GPU memory: 24GB(RTX 3090)  
CPU memory: 128GB  
The system will work in the following environment.  
If you have less than that, it will probably stop working in places, so please adjust the code accordingly.
