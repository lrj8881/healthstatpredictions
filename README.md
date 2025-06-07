Using Health Data for Mortality Prediction with Pytorch and SQLite

Elizabeth Javor, elizabethjavor@proton.me
-----------------------------------------------------------------------
This project contains code that uses a neural net and linear regression to predict the deaths per 
million from diabetes and heart disease in a specific state and a specific year using from various 
physical health statistics of the population. 

The data is of all 50 states over the years 2011-2017 (so one data point corresponds to a state and a year.)

A more thorough writeup (and results) can be found at Health Statistics as Mortality Predictors.pdf.
This README details how to run the code. 

Codebase:
Health Statistics as Mortality Predictors.pdf: A writeup of this project and its outcomes. 
models.py: The command line interface, and the models (neural net, linear)
preprocessing.py: Converts csv files stored in "data" folder to an SQLite database
dataset.py: creates pytorch dataset from SQLite database
helperfunctions.py: Graphing functions and SQLite helper functions
README.md: This file.
data (folder): Contains all data files necessary; sources can be found in the writeup pdf. 
Contents:
--NCHS_-_Leading_Causes_of_Death__United_States.csv 
--Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv
--historical_state_population_by_year.csv
logs(folder): Contains example log files produced by grid search, the ones used to fill the default dictionary (see description for the --gs argument below). (If gridsearch is run, this is where new logs
will go).
Contents:
--diabetes_linear_Batch32AdamMSE_FEATS_1_2_3_4_5_6.log
--diabetes_nn_Batch32AdamMSE_FEATS_1_2_3_4_5_6.log
--heart disease_linear_Batch32AdamMSE_FEATS_1_2_3_4_5_6.log
--heart disease_nn_Batch32AdamMSE_FEATS_1_2_3_4_5_6.log

Dependencies:

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Running the Code:

To run, use the command
python models.py <cause> <modeltype> [--features 1 2 3 4 5 6] [--gs N] [-g] [-s]

<cause> can theoretically be any cause appearing in the dataset "C:\Users\drago\Downloads\NCHS_-_Leading_Causes_of_Death__United_States.csv"
but code has been tested on and built for the two causes of death "diabetes" and "heart disease"

<modeltype> can be either linear or NN (neural net)

--features:

There are 5 viable features to include, which are (for a state and a year) the following statistics:

    1 Percent of adults aged 18 years and older who have obesity
    2 Percent of adults who engage in no leisure-time physical activity
    3 Percent of adults who engage in muscle-strengthening activities on 2 or more days a week
    4 Percent of adults who achieve at least 150 minutes a week of moderate-intensity aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic activity
    5 Percent of adults who achieve at least 300 minutes a week of moderate-intensity aerobic activity or 150 minutes a week of vigorous activity

Default includes all features. Use a space-separated list to specify features. 

--gs turns on grid search, which is done over the following lists of hyperparameters:
Epochs: [500, 1000, 2000]
Train/Validation splits: [0.75, 0.8, 0.9]
Learning rates: [0.01, 0.001, 0.0001]

As its argument, --gs takes in an integer for the number of gridsearch trials. Gridsearch outputs a .log file of the name <cause><modeltype>Batch32AdamMSE_FEATS<features>.log in the "logs" folder.

(When not activated (or set to 0), simply uses best hyperparameters for model (preprogrammed) from my own grid search on all features for each model type and cause.)

-g1 flag turns on feature vs prediction graphing (in matplotlib). Shows a scatterplot of the data for each feature and a red line for the model prediction.

-g2 flag turns on actual vs prediction graphing. Shows a scatterplot of actual vs. predicted DPM and
a red line for a perfect regression model.

-s flag turns on saving of the model (Folder "model" will be created on first use.)

EXAMPLE:

python models.py "Heart Disease" NN --features 1 2 4 5 --gs 3 -g2

will train a neural network to predict the deaths per million from heart disease using features 1, 2, 4, and 5, perform grid search over 3 trials (and save the best params in the SQL database), and graph
actual vs predicted DPM for the best performing model

