# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This repo is designed to read churn data and build a churn classification model.
It also consists of a file that can be used to check the churn modelling module.
For information read below.

## Files and data description

The file Structure looks as follows:
```bash
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── conftest.py
├── constants.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── __init__.py
├── LICENSE
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── plots
│   ├── eda_plots
│   │   ├── Attrition_Flag_hist.png
│   │   ├── correlation_plot.png
│   │   ├── Customer_Age_hist.png
│   │   ├── Marital_Status_bar.png
│   │   └── Total_Trans_Ct_density.png
│   └── results
│       ├── Logistic Regression_classification_report.png
│       ├── Random Forest_classification_report.png
│       ├── rf_feature_importance_plot.png
│       └── roc_plots.png
├── pyproject.toml
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt

```



### 1. churn_library.py
Overview of the files and data present in the root directory: \
The churn model is trained in the **churn_library.py** module. This module is responsible for reading the data, generating EDA plots, doing feature engineering, building the model, saving the model and saving the feature importance and roc plots. \
The models are saved at **'./models/'** . \
The EDA plots are saved at **'./plots/eda_plots/'** . \
The roc curve, feature importance plot and classification reports are saved at **'./plots/results/'** . \
Python Libraries for running the module are ```os, numpy, pandas, matplotlib, seaborn, sklearn, joblib```

### 2. churn_notebook.ipynb
Is the notebok based on which the **churn_library.py** module was created. And has all the code present in churn_library.py but in unstructured manner. \
Python Libraries for running the module are ```os, numpy, pandas, matplotlib, seaborn, sklearn, joblib, shap```

### 3. churn_script_logging_and_tests.py
Runs the tests on churn_library.py module using pytest. It check for data ingestion, feature encoding, feature engineering and if all the plots are generated and saved at the expected location.
The logs generated from this file are saved in **'./logs/churn_library.log'**
The configurations required for logging are done in **conftest.py** which creates a fixture called logger. It specifies the filename where logs will be saved, level of logging etc.
Python Libraries for running the module are ```os, logging, pandas```

### 4. conftest.py
The configurations required for logging are done in **conftest.py** which creates a fixture called logger. It specifies the filename where logs will be saved, level of logging etc.
Python Libraries for running the module are ```logging, pytest```

### 5. constants.py
This file saves all the constants and paths that are used by both **churn_library.py** and **churn_script_logging_and_tests.py**.

### 6.data/
The folder that contanis the data for training the model in csv format - **bank_data.csv**

### 7. Guide.ipynb
The guide used for creating this project. Contains details on specifications to follow.

### 8. logs/
The folder contains the log file generated after running the testing script **churn_script_logging_and_tests.py**
The log file contains info level information of the code and can be used to check status of different components.

### 9. models/
Contains models trained by **churn_library.py** \
&nbsp;&nbsp;&nbsp;&nbsp;1. logistic_model.pkl --> The trained logistic regression model \
&nbsp;&nbsp;&nbsp;&nbsp;2. rfc_model.pkl --> The trained random forest model \

### 10. plots
Contains eda(exploratory) and model result plots created by the model \
&nbsp;&nbsp;&nbsp;&nbsp;1. eda_plots \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.Attrition_Flag_hist.png  --> Histogram for the Attrition Flag \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. correlation_plot.png  --> The plot showing correlation between different features \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Customer_Age_hist.png  --> HIstogram for Cutomer Ages feature \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Marital_Status_bar.png  --> Bar Chart for the Martial Status feature \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. Total_Trans_Ct_density.png  --> Density plot for Total Trans Ct \
&nbsp;&nbsp;&nbsp;&nbsp;2. results \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Logistic Regression_classification_report.png  --> Classification report for logistic regression model \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Random Forest_classification_report.png  --> Classification report for random forest model \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. rf_feature_importance_plot.png --> PLot showing imporatnce of features based on Random Forest \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. roc_plots.png --> Receiver operator curve comparing Logistic Regression and Random Forest model \

### 11. pyproject.toml
File for configuring the behavior of testing with pytest. It is used here to specify to pytest that logging should be done in a file and not printed on the terminal.

### 12. Requirements Files
requirements_py3.6.txt --> Python dependencies for version 3.6
requirements_py3.8.txt --> Python dependencies for version 3.8

## Running Files
How do you run your files? What should happen when you run your files?

### churn_script_logging_and_tests.py
For testing the churn_library.py run following command in the terminal after you are at root of the churn_prediction library: 

     `> pytest churn_script_logging_and_tests.py`

This should create a **'./logs/churn_library.log'** file with all the logs related to the test.

### churn_library.py
For building the model and the related plots run the following command in the terminal after you are at root of the churn_prediction library:

    `> python churn_library.py`

This will created the models and save it along with the plots in the paths mentioned above.



