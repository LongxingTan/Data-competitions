import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=200):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_rmse(y_true, y_pred, transform=None):
    if transform is not None:
        y_pred = transform(y_pred)  
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def coef(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = np.sum((x - x_mean) * (y - y_mean))
    c2 = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)) + 1e-10
    return c1 / c2


def eval_score(y_true, y_pred, transform=None):
    # overall score is: np.sum()
    if transform is not None:
        y_pred = transform(y_pred)  
        
    accskill_score = []
    rmse_score = []
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    predict_sequence_length = y_true.shape[-1]

    for i in range(predict_sequence_length):
        rmse_i = eval_rmse(y_true[:, i], y_pred[:, i])
        cor_i = coef(y_true[:, i], y_pred[:, i])

        accskill_score.append(a[i] * np.log(i+1) * cor_i)
        rmse_score.append(rmse_i)

    accskill_score = np.array(accskill_score)
    rmse_score = np.array(rmse_score)    

    return  np.sum(2 / 3.0 * accskill_score - rmse_score)


def custome_rmse_fn(y_true, y_pred):
    """ custome loss function
    The 24 series is not equally weighted, so log1p weighted is used.
    This is just my initial try, still have further improvement space.

    y_true: batch * 24
    """ 
    diff = (y_pred - y_true) ** 2
    predict_sequence_length = tf.shape(y_true)[-1]
    alpha = [np.log1p(i) for i in range(1, predict_sequence_length+1)]
    #alpha = [np.log(i)*j for i,j in zip(range(1, predict_sequence_length+1), [0.65]*4+[1]*7+[1.2]*7+[1.5]*6)]
    alpha = tf.reshape(tf.convert_to_tensor(alpha, tf.float32), (1, predict_sequence_length))
    rmse = tf.sqrt(tf.reduce_mean(diff * alpha))
    return rmse
