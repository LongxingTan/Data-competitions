#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, www.yuetan.space
# @date: 2021-11
# This script defines the basic trainer for tabular data/ time series data


import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier, Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as colormap


class Trainer(object):
    """ A trainer for GBDT methods
    # how to use it:
    model = Trainer(CatBoostClassifier(**CAT_PARAMS))
    model.train(x_train, y_train, x_valid, y_valid, fit_params={}, importance_method='auto')
    """

    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__

    def train(self, x_train, y_train, x_valid=None, y_valid=None, categorical_feature=None, fit_params=None,
              importance_method='auto'):
        self.input_shape = x_train.shape

        # LightGBM 模型
        if self.model_type[:4] == 'LGBM':
            if x_valid is not None:
                self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                               categorical_feature=categorical_feature, **fit_params)
            else:
                self.model.fit(x_train, y_train, eval_set=[(x_train, y_train)],
                               categorical_feature=categorical_feature, **fit_params)

            self.best_iteration = self.model.best_iteration_

        self.imps = self.create_feature_importance(use_method=importance_method)

    def predict(self, x_test, method='predict', num_iteration=None):
        if method == 'predict':
            if num_iteration:
                return self.model.predict(x_test, num_iteration=num_iteration)
            else:
                return self.model.predict(x_test)
        elif method == 'predict_proba':
            if num_iteration is not None:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)
            else:
                return self.model.predict_proba(x_test)
        elif method == 'predict_proba_positive':
            if num_iteration is not None:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)[:, 1]
            else:
                return self.model.predict_proba(x_test)[:, 1]
        else:
            raise ValueError("unsupported predict method of {}".format(method))

    def create_feature_importance(self, use_method='auto', importance_params={}):
        if use_method == "auto":
            # split and gain
            return self.model.feature_importances_

    def get_feature_importance(self, columns=None, save=False, save_dir='./feature_importance.csv', plot=False):
        if columns is None or len(columns) != len(self.imps):
            columns = ['feature_{}'.format(i) for i in range(len(self.imps))]

        imps = pd.DataFrame(self.imps, index=columns)

        if save:
            imps.sort_values(0, ascending=False, inplace=True)
            imps.to_csv(save_dir, index=True)

        if plot:
            plt.figure(figsize=(5, int(len(columns) / 3)))
            imps_mean = np.mean(self.imps.values, axis=1)
            imps_se = np.std(self.imps, axis=1) / np.sqrt(self.imps.shape[0])
            order = np.argsort(imps_mean)
            colors = colormap.winter(np.arange(len(columns)) / len(columns))
            plt.barh(np.array(columns)[order],
                     imps_mean[order], xerr=imps_se[order], color=colors)
            plt.show()
        return imps

    def plot_feature_importance(self, columns=None, use_method='auto'):
        imps = self.get_feature_importance(use_method)

        if columns is None:
            columns = ['feature_{}'.format(i) for i in range(len(imps))]

        plt.figure(figsize=(5, int(len(columns) / 3)))
        order = np.argsort(imps)
        colors = colormap.winter(np.arange(len(columns)) / len(columns))
        plt.barh(np.array(columns)[order], imps[order], color=colors)
        plt.show()

    def get_model(self):
        return self.model

    def save_model(self, model_dir):
        joblib.dump(self.model, model_dir)
        return

    def get_best_iteration(self):
        return self.best_iteration
