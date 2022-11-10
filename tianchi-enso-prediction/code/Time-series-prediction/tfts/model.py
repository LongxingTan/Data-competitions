#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,  ModelCheckpoint, TensorBoard
from .models import Seq2Seq


def build_tfts_model(use_model='seq2seq', custom_model_params={}, dynamic_decoding=True):
    if use_model == 'seq2seq':
        Model = Seq2Seq(custom_model_params=custom_model_params, dynamic_decoding=dynamic_decoding)
    return Model
