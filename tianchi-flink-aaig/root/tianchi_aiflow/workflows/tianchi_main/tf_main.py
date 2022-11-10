#coding:utf-8

"""
基本逻辑： 由tianchi_main中的run_tianchi_project 调用 tianchi_executor中的TrainModel, 调用tf_main中的train
日期:2021-11-09 12:53:41

score:1.4623
F1_0:0.7316
valid_latency_0:0.9968
F1_1:0.7331
valid_latency_1:1.0000
"""

import os
import gc
import logging
import datetime
import json
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

os.environ["OMP_NUM_THREADS"] = str(6)


# =============================== Model =====================================

class MLP(object):
    def __init__(self):
        self.dense1 = Dense(64, use_bias=True, activation='tanh')
        self.dense2 = Dense(32, use_bias=True, activation='tanh')
        self.dense3 = Dense(64, use_bias=True, activation='tanh')
        self.dense4 = Dense(64, use_bias=True, activation='tanh')

        self.dense8 = Dense(64, use_bias=True, activation='tanh')
        self.dense9 = Dense(128, use_bias=True, activation='tanh') 
        self.dense10 = Dense(128, use_bias=True, activation='tanh')
        self.dense11 = Dense(64, use_bias=True, activation='tanh') 
        self.dense12 = Dense(64, use_bias=True, activation='tanh')

        self.dense0 = Dense(units=32, activation='tanh')
        self.dense = Dense(units=1, activation='sigmoid')
        self.drop1 = Dropout(0.125)
        self.drop2 = Dropout(0.24)
        self.bn1 = BatchNormalization()

    def __call__(self, x):
        x = self.bn1(x)
        x = self.drop1(x)
        x1 = x[:, :72]
        x3 = x[:, 72:]

        x1 = self.dense1(x1)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        x1 = self.dense4(x1)
        
        x3 = self.dense8(x3)
        x3 = self.dense9(x3)   
        x3 = self.dense10(x3)   
        x3 = self.drop2(x3)  
        x3 = self.dense11(x3)
        x3 = self.dense12(x3)

        y = self.dense0(x1 + x3)
        y = self.dense(y)
        return y


def build_model(num_classes=2, threshold=0.45):
    inputs = (Input(shape=152, name='input'))  # stage II
    outputs = MLP()(inputs)

    pred_label = tf.cast(outputs > threshold, tf.int32, name='pred_label')
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


# =============================== Data =====================================

def build_data(train_path, neg_sample=False, neg_sample2=True, neg_sample3=False, ignore_cold=False, augment_data=True, augment_stage2=False):
    data = pd.read_csv(train_path, header=None)
    unlabeled_data = data.loc[data[5] < 0].copy()
    data = data.loc[data[5] >= 0]  # choose the labeled data
    data.sort_values(by=1, ascending=True, inplace=True)  # visit time
    data.drop_duplicates(subset=[2, 3, 4, 5], keep='last', inplace=True)  # remove duplicates

    if ignore_cold:  # cold start user
        feature = data[4].str.split(" ", expand=True).astype(np.float32)
        data = data.loc[feature.iloc[:, 72:].sum(axis=1) > 0]

    if neg_sample:  # negative sample resampling
        data_pos = data[data[5] == 1]
        data_neg = data[data[5] == 0]
        data_neg = data_neg.sample(frac=0.8, random_state=315, replace=False)
        data = pd.concat([data_pos, data_neg])
    
    if neg_sample2:  # negative sample choose by user_id
        data_pos = data[data[5] == 1]
        data_neg = data[data[5] == 0]
        data_neg = data_neg.drop_duplicates(subset=[2], keep='last')

        if neg_sample3:  # choose by ratio
            white_list = np.unique(data_pos[2].tolist())
            data_neg = data_neg.loc[~data_neg[2].isin(white_list)]   
        data = pd.concat([data_pos, data_neg])
    
    if augment_data:  # positive sample augmentation
        data_stat = data.groupby([2]).agg({5: ['count', 'mean']}).reset_index()
        data_stat.columns = data_stat.columns.get_level_values(1)
        data_stat.sort_values('count', ascending=False, inplace=True)
        white_list = data_stat.loc[(data_stat['count'] > 2) & (data_stat['mean'] > 0.88)].iloc[:, 0].to_list()

        guess_data = unlabeled_data.loc[unlabeled_data[2].isin(white_list)]
        guess_data = guess_data.sample(frac=1.)
        guess_data['cumcount'] = guess_data.groupby([2])[5].cumcount()  
        guess_data[5] = 1
        guess_data = guess_data.loc[guess_data['cumcount'] < 6]
        guess_data.drop(['cumcount'], axis=1)
        data = pd.concat([data, guess_data])
    
    if augment_stage2:  # which means augment by using train0.csv
        history_data = pd.read_csv(train_path.replace('train1.csv', 'train0.csv'), header=None)
        history_data_pos = history_data.loc[history_data[5] > 0]  # select positive
        history_data_pos.drop_duplicates(subset=[2], keep='last')  # keep unique user_id
        data = pd.concat([data, history_data_pos])

    x = data[4].str.split(" ", expand=True).values.astype(np.float32)
    y = data[5].values
    print("data prepared done")
    return x, y


# =============================== Train =====================================

def scheduler(epoch):
    if epoch < 10:
        return 3.3e-3
    elif epoch <= 45:
        return 2.1e-4
    else:
        return 1e-5


def scheduler2(epoch):
    if epoch <= 25:
        return 2e-4
    else:
        return 1e-5


def train(train_path, model_dir, save_name):
    """
    input：train_path, model_dir, save_name
    output：save pb model to model_dir + "/frozen_model"
    """
    set_seed(1988)
    fit_params = {
        'n_epochs': 60,
        'batch_size': 1024,
        'learning_rate': 2.1e-4,
        'threshold': 0.391,
    }
    
    model = build_model(threshold=fit_params['threshold'])
    save_path = os.path.join(model_dir, save_name)

    if not os.path.exists(save_path + '.meta'):
        # Stage I
        x, y = build_data(train_path)
        lrs = LearningRateScheduler(schedule=scheduler)
    else: 
        # Stage II，load pretrained model
        x, y = build_data(train_path, augment_stage2=False)
        sess = tf.keras.backend.get_session()
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        # Stage II，update training parameters
        fit_params.update({'n_epochs': 35, 'learning_rate': 2e-4, 'threshold': 0.391})
        lrs = LearningRateScheduler(schedule=scheduler2)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.022)
    model.compile(tf.keras.optimizers.Adam(fit_params['learning_rate']), loss, ['accuracy'])

    os.makedirs(model_dir, exist_ok=True)    
    model.fit(x, y, batch_size=fit_params['batch_size'], epochs=fit_params['n_epochs'], verbose=2, shuffle=True, class_weight={0: 0.95, 1: 1}, callbacks=[lrs])
    # save the model
    output_model_dir = os.path.join(model_dir, "frozen_model")
    model_name = 'frozen_inference_graph.pb'
    sess = tf.keras.backend.get_session()

    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(model_dir, save_name))
    print('\n model saved in {}'.format(save_path))

    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, ['pred_label'])
    tf.train.write_graph(output_graph_def, output_model_dir, model_name, as_text=False)
    input_names_new = [i + ':0' for i in ['input']]
    output_names_new = [o + ':0' for o in ['pred_label']]
    meta = {
        "input_names": input_names_new,
        "output_names": output_names_new
    }

    with open(os.path.join(output_model_dir, "graph_meta.json"), "w") as f:
        f.write(json.dumps(meta))
    print('\npb model saved successfully :)')
    sess.close()
    tf.keras.backend.clear_session()
    del x, y
    gc.collect()


def save_model(input_model_dir, output_model_dir, model_name):
    # input_model = tf.keras.models.load_model(input_model_dir)

    sess = tf.keras.backend.get_session()
    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, ['pred_label'])
    tf.train.write_graph(output_graph_def, output_model_dir, model_name, as_text=False)

    input_names_new = [i + ':0' for i in ['input']]
    output_names_new = [o + ':0' for o in ['pred_label']]
    meta = {
        "input_names": input_names_new,
        "output_names": output_names_new
    }

    with open(os.path.join(output_model_dir, "graph_meta.json"), "w") as f:
        f.write(json.dumps(meta))
    print('\nmodel saved successfully :)')


class Evaluator(tf.keras.callbacks.Callback):
    """
    evaluator = Evaluator()
    model.fit(training_data, training_target, validation_data=(validation_data, validation_target), nb_epoch=10, batch_size=64, callbacks=[evaluator])
    """
    def __init__(self, x_valid, y_valid) -> None:
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        y_valid_pred = (np.asarray(self.model.predict(self.x_valid))).round()    
        val_f1 = f1_score(self.y_valid, y_valid_pred)
        val_recall = recall_score(self.y_valid, y_valid_pred)
        val_precision = precision_score(self.y_valid, y_valid_pred)
        print(" — val_f1: %f — val_precision: %f — val_recall %f " % (val_f1, val_precision, val_recall))


def set_seed(seed=315):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    base_dir = '/home/tlx/competition/AAIG2/tcdata'
    train_path = base_dir + '/train0.csv'
    model_dir = '/home/tlx/competition/AAIG2/weights' + '/model/base_model'
    save_name = 'base_model'

    train(train_path, model_dir, save_name)
