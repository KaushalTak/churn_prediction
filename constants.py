DATA_PATH = "./data/bank_data.csv"
EDA_PLOTS_PATH = "./plots/eda_plots/"
MODEL_RESULT_PLOTS_PATH = "./plots/results/"
MODEL_SAVE_PATH = "./models/"

LABEL = 'Attrition_Flag'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]


PLOT_SIZE = (20, 10)

HIST_PLOT_COLUMNS = ['Attrition_Flag', 'Customer_Age']
BAR_PLOT_COLUMNS = ['Marital_Status']
DENSITY_PLOT_COLUMNS = ['Total_Trans_Ct']

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Attrition_Flag', 'Education_Level_Attrition_Flag', 'Marital_Status_Attrition_Flag',
             'Income_Category_Attrition_Flag', 'Card_Category_Attrition_Flag']

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

EXPECTED_MODEL_SAVED = ['logistic_model.pkl', 'rfc_model.pkl']

EXPECTED_MODEL_RESULTS_SAVED = ['Logistic Regression_classification_report.png',
                                'Random Forest_classification_report.png',
                                'rf_feature_importance_plot.png', 'roc_plots.png']
