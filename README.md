# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This repo is designed to read churn data and build a churn classification model.
It also consists of a file that can be used to check the churn modelling module.
For information read below.

## Files and data description
### churn_library.py
Overview of the files and data present in the root directory:
The churn model is trained in the `churn_library.py` module. This module is responsible for reading the data, generating EDA plots, doing feature engineering, building the model, saving the model and saving the feature importance and roc plots.
The models are saved at **'./models/'** .
The EDA plots are saved at **'./plots/eda_plots/'** .
The roc curve, feature importance plot and classification reports are saved at **'./plots/results/'** .

### churn_script_logging_and_tests.py
Runs the tests on churn_library.py module using pytest. It check for data ingestion, feature encoding, feature engineering and if all the plots are generated and saved at the expected location.
The logs generated from this file are saved in **'./logs/churn_library.log'**
The configurations required for logging are done in **conftest.py** which creates a fixture called logger. It specifies the filename where logs will be saved, level of logging etc.

### constants.py
This file saves all the constants and paths that are used by both **churn_library.py** and **churn_script_logging_and_tests.py**.

## Running Files
How do you run your files? What should happen when you run your files?

### churn_script_logging_and_tests.py
For testing the churn_library.py run following command in the terminal after you are at root of the churn_prediction library: 
     > pytest churn_script_logging_and_tests.py
This should create a **'./logs/churn_library.log'** file with all the logs related to the test.

### churn_library.py
For building the model and the related plots run the following command in the terminal after you are at root of the churn_prediction library:
    > python churn_library.py
This will created the models and save it along with the plots in the paths mentioned above.



