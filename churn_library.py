'''
Title: Module to train Logistic and Random Forest Model for Churn/Attrition Prediction

Author: Tak
Date: April 26, 2022
'''

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import joblib
from constants import CAT_COLUMNS, KEEP_COLS, LABEL, PARAM_GRID
from constants import PLOT_SIZE, HIST_PLOT_COLUMNS, BAR_PLOT_COLUMNS, DENSITY_PLOT_COLUMNS
from constants import DATA_PATH, EDA_PLOTS_PATH, MODEL_RESULT_PLOTS_PATH, MODEL_SAVE_PATH


sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.figure(figsize=PLOT_SIZE)
fig, ax = plt.subplots()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    data_frame[LABEL] = data_frame[LABEL].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


def plot_hist_save(data_frame, column_name):
    '''
    helper function to create and save histogram
    input:
            data_frame: dataframe with data
            column_name: column for which histogram is created
    output:
            None
    '''
    fig, ax = plt.subplots()
    data_frame.hist(column_name, ax=ax)
    fig.savefig('{}{}_hist.png'.format(EDA_PLOTS_PATH, column_name),
                bbox_inches="tight")


def plot_bar_save(data_frame, column_name):
    '''
    helper function to create and save bar chart
    input:
            data_frame: dataframe with data
            column_name: column for which bar chart is created
    output:
            None
    '''
    fig, ax = plt.subplots()
    bar_chart = data_frame[column_name].value_counts(
        'normalize').plot(kind='bar')
    bar_chart.figure.savefig('{}{}_bar.png'.format(EDA_PLOTS_PATH, column_name),
                             bbox_inches="tight")


def plot_density_plot(data_frame, column_name):
    '''
    helper function to create and save density plot
    input:
            data_frame: dataframe with data
            column_name: column for which density plot is created
    output:
            None
    '''
    fig, ax = plt.subplots()
    sns_fig = sns.histplot(
        data_frame['Total_Trans_Ct'], stat='density', kde=True)
    sns_fig.figure.savefig(
        '{}{}_density.png'.format(EDA_PLOTS_PATH, column_name), bbox_inches="tight")


def correlation_heat_map(data_frame):
    '''
    helper function to create and save correlation heatmap
    input:
            data_frame: dataframe with data
    output:
            None
    '''
    sns_fig = sns.heatmap(data_frame.corr(), annot=False,
                          cmap='Dark2_r', linewidths=2)
    sns_fig.figure.savefig(
        '{}correlation_plot.png'.format(EDA_PLOTS_PATH), bbox_inches="tight")


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    for col in HIST_PLOT_COLUMNS:
        plot_hist_save(data_frame, col)
    for col in BAR_PLOT_COLUMNS:
        plot_bar_save(data_frame, col)
    for col in DENSITY_PLOT_COLUMNS:
        plot_density_plot(data_frame, col)
    correlation_heat_map(data_frame)


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                      be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for feat in category_lst:
        temp = []
        groups = data_frame.groupby(feat)[response].mean()
        for val in data_frame[feat]:
            temp.append(groups.loc[val])
        data_frame['{}_{}'.format(feat, response)] = temp
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument
                        that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    data_frame = encoder_helper(data_frame, CAT_COLUMNS, response)
    feature_df = data_frame[KEEP_COLS]
    y_labels = data_frame[response]
    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, y_labels, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def _plot_and_save_classification_report(y_train, y_test, y_train_preds, y_test_preds, model_name):
    '''
    helper function to create and save classification report
    input:
            y_train: y training data
            y_test: y testing data
            y_train_preds: training predictions from logistic regression
            y_test_preds: test predictions from logistic regression
            model_name: name of the model which made prediction
    output:
            None
    '''
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('{} Train'.format(model_name)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('{} Test'.format(model_name)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        '{}{}_classification_report.png'.format(
            MODEL_RESULT_PLOTS_PATH, model_name),
        bbox_inches='tight')


def _plot_and_save_roc_curves(lr_model, rfc_model, x_test, y_test):
    '''
    helper function to plot and save roc curves for random forest and logistic
    regression classifiers
    input:
            lr_model: logistic regression model
            rfc_model: random forest model
            x_test: testing feature data
            y_test: test labels
    output:
            None
    '''
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('{}roc_plots.png'.format(
        MODEL_RESULT_PLOTS_PATH), bbox_inches="tight")


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    _plot_and_save_classification_report(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, model_name='Random Forest')
    _plot_and_save_classification_report(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr, model_name='Logistic Regression')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth,
                bbox_inches='tight')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)
    cv_rfc.fit(x_train, y_train)
    lr_model.fit(x_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    joblib.dump(cv_rfc.best_estimator_,
                '{}rfc_model.pkl'.format(MODEL_SAVE_PATH))
    joblib.dump(lr_model, '{}logistic_model.pkl'.format(MODEL_SAVE_PATH))
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)
    _plot_and_save_roc_curves(lr_model, rfc_model, x_test, y_test)
    feature_importance_plot(
        rfc_model, x_train, '{}rf_feature_importance_plot.png'.format(MODEL_RESULT_PLOTS_PATH))
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)


if __name__ == '__main__':
    data = import_data(DATA_PATH)
    print('Data import completed!')
    perform_eda(data)
    print('EDA done!')
    x_training, x_testing, y_training, y_testing = perform_feature_engineering(
        data, LABEL)
    print('Feature Engineering completed!')
    train_models(x_training, x_testing, y_training, y_testing)
    print('Model trained and saved!')
