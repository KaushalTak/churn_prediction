'''
Module to test the functions in churn_library.py

Author: Tak
Date: April 26, 2022
'''

import os
import logging
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper
from churn_library import perform_feature_engineering, train_models
from constants import HIST_PLOT_COLUMNS, BAR_PLOT_COLUMNS, DENSITY_PLOT_COLUMNS
from constants import CAT_COLUMNS, LABEL
from constants import DATA_PATH, EDA_PLOTS_PATH, MODEL_RESULT_PLOTS_PATH, MODEL_SAVE_PATH
from constants import EXPECTED_MODEL_SAVED, EXPECTED_MODEL_RESULTS_SAVED


def test_import(request, logger):
    '''
    test data import
    '''
    try:
        data = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
        request.config.cache.set(
            'cache_data', data.to_dict('list'))
    except FileNotFoundError as err:
        logging.error("Testing import data: The file wasn't found")
        raise err
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info('Testing import_data: File has rows and columns')
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(logger, request):
    '''
    test perform eda function
    '''
    try:
        data = request.config.cache.get('cache_data', None)
        data = pd.DataFrame(data)
        data.drop('Unnamed: 0', axis=1, inplace=True)
        logging.info(
            "Testing perform_eda: SUCCESS - Received dataframe from cache")
        perform_eda(data)
        expected_files = []
        expected_files += ['{}_hist.png'.format(feat)
                           for feat in HIST_PLOT_COLUMNS]
        expected_files += ['{}_bar.png'.format(feat)
                           for feat in BAR_PLOT_COLUMNS]
        expected_files += ['{}_density.png'.format(feat)
                           for feat in DENSITY_PLOT_COLUMNS]
        expected_files.append('correlation_plot.png')
        outputed_eda_files = os.listdir(EDA_PLOTS_PATH)
        assert set(outputed_eda_files) == set(expected_files)
        logging.info(
            'Testing perform_eda: SUCCESS - All EDA files generated')
    except AssertionError as err:
        logging.error(
            'Testing perform_eda: FAILED - EDA files different from expected')
        raise err


def test_encoder_helper(logger, request):
    '''
    test encoder helper
    '''
    try:
        data = request.config.cache.get('cache_data', None)
        data = pd.DataFrame(data)
        data.drop('Unnamed: 0', axis=1, inplace=True)
        logging.info(
            "Testing encoder_helper: SUCCESS - Received dataframe from cache")
        data = encoder_helper(data, CAT_COLUMNS, LABEL)
        logging.info(
            'Testing encoder_helper: SUCCESS - encoder_helper ran without error')
        expected_cols = ['{}_{}'.format(feat, LABEL) for feat in CAT_COLUMNS]
        assert len(set(expected_cols).difference(set(data.columns))) == 0
        logging.info(
            "Testing encoder_helper: SUCCESS - encoder_helper created all columns")
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: FAILED - encoder_helper \
            created different columns then expected')
        raise err


def test_perform_feature_engineering(logger, request):
    '''
    test perform_feature_engineering
    '''
    try:
        data = request.config.cache.get('cache_data', None)
        data = pd.DataFrame(data)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - Received dataframe from cache")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data, LABEL)
        assert x_train.shape[0] > x_test.shape[0] > 0
        assert x_train.shape[1] == x_train.shape[1] > 0
        assert len(y_train) > len(y_test) > 0
        request.config.cache.set(
            'x_train', x_train.to_dict('list'))
        request.config.cache.set(
            'x_test', x_test.to_dict('list'))
        request.config.cache.set(
            'y_train', list(y_train))
        request.config.cache.set(
            'y_test', list(y_test))
        logging.info(
            'Testing perform_feature_engineering: SUCCESS - splitted data passed sanity check')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: FAILED - splitted data failed sanity check')
        raise err


def test_train_models(logger, request):
    '''
    test train_models
    '''
    try:
        x_train = request.config.cache.get('x_train', None)
        x_test = request.config.cache.get('x_test', None)
        y_train = request.config.cache.get('y_train', None)
        y_test = request.config.cache.get('y_test', None)
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
        logging.info(
            'Testing train_models: SUCCESS, got training testing data')
        train_models(x_train, x_test, y_train, y_test)
        logging.info(
            'Testing train_models: SUCCESS, train_models function ran without error')
        plots = os.listdir(MODEL_RESULT_PLOTS_PATH)
        models = os.listdir(MODEL_SAVE_PATH)
        assert set(models) == set(EXPECTED_MODEL_SAVED)
        assert set(plots) == set(EXPECTED_MODEL_RESULTS_SAVED)
        logging.info(
            'Testing train_models: SUCCESS - models and plots generated.')
    except AssertionError as err:
        logging.error(
            'Testing train_models: FAILED - models and plots not as expected.')
        raise err
