#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, www.yuetan.space
# @date: 2021-07
# This script defines the basic trainer for tabular data by sklearn


from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC


class SKTrainer(object):
    def __init__(self, model) -> None:
        self.model = model

    def train(self, x_train, y_train, fit_params=None):
        self.model.fit(x_train, y_train, **fit_params)
