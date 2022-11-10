# A general trainer for TensorFlow2, 

import numpy as np
import tensorflow as tf
from copy import copy, deepcopy

__all__ = 'Trainer'
__author__ = '公众号YueTan'


class Trainer(object):
    def __init__(self, model, loss, optimizer, lr_scheduler=None, metrics=None):
        # model: a tf.keras.Model instance
        # loss: a loss function
        # optimizer: tf.keras.Optimizer instance
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics

    def train(self, train_dataset, valid_dataset=None, n_epochs=10, batch_size=8, learning_rate=3e-4, verbose=2, eval_metric=(), model_dir=None, use_ema=False, stop_no_improve_epochs=None, transform=None):
        # train_dataset: tf.data.Dataset instance
        # valid_dataset: None or tf.data.Dataset instance
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.eval_metric = eval_metric
        self.use_ema = use_ema
        self.transform = transform

        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(0.9).apply(self.model.trainable_variables)
        
        # early stop, socre "higher is better"
        if stop_no_improve_epochs is not None:
            no_improve_epochs = 0
            best_metric = -np.inf

        # model_dir
        if model_dir is None:
            model_dir='../user_data/nn'

        for epoch in range(n_epochs):
            epoch_loss, scores = self.train_loop(train_dataset)

            if valid_dataset is not None:
                valid_epoch_loss, valid_scores = self.valid_loop(valid_dataset)
            else:  # Temp, 针对特殊的验证方式
                valid_scores = 0 #soda_valid()

            print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, epoch_loss, valid_epoch_loss) + ','.join([' Valid Metrics{}: {:.4f}'.format(i, me) for i, me in enumerate(valid_scores)]) ) #, Valid Score: {:.4f}, Valid RMSE: {:.4f}'.format(epoch+1, epoch_loss, valid_epoch_loss, np.sum(valid_scores[0]), np.mean(valid_scores[1])))
            #print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, epoch_loss) + ','.join([' Valid Metrics{}: {:.4f}'.format(i, me) for i, me in enumerate(valid_scores)]) ) #, Valid Score: {:.4f}, Valid RMSE: {:.4f}'.format(epoch+1, epoch_loss, valid_epoch_loss, np.sum(valid_scores[0]), np.mean(valid_scores[1])))
            if epoch >= 4:  # train at least 5 epochs
                if stop_no_improve_epochs is not None:  # if activate early stop
                    if valid_scores[0] >= best_metric:
                        #self.export_model(model_dir)
                        best_metric = valid_scores[0]
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        # self.learning_rate = self.learning_rate / 2
                    if no_improve_epochs >= stop_no_improve_epochs:  
                        print('I have tried my best, no improved and stop training!')
                        break   

        # 保存模型
        self.export_model(model_dir, only_pb=True)     
            
    
    def train_loop(self, dataset):
        epoch_loss = 0.
        y_trues, y_preds = [], []

        for step, (x_train, y_train) in enumerate(dataset):
            y_pred, loss = self.train_step(x_train, y_train)
            epoch_loss += tf.reduce_mean(loss)  # for each sample
            y_trues.append(y_train)
            y_preds.append(y_pred)

        scores = []
        if self.eval_metric:        
            y_preds = tf.concat(y_preds, axis=0)
            y_trues = tf.concat(y_trues, axis=0)
            
            for metric in self.eval_metric:
                scores.append(metric(y_trues, y_preds, transform=self.transform))
        return epoch_loss/(step+1), scores            

    def train_step(self, x_train, y_train):
        # train_step for one batch
        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)            
            loss = self.loss_fn(y_train, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -5.0, 5.0)) for grad in gradients]  # gradient clip
        opt = self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.use_ema:
            with tf.control_dependencies([opt]):
                ema_op = self.ema

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.global_step)
            #print(lr)
        else:
            lr = self.learning_rate
        self.optimizer.lr.assign(lr)
        self.global_step.assign_add(1)        
        return y_pred, loss

    def valid_loop(self, valid_dataset):
        valid_epoch_loss = 0
        y_valid_trues, y_valid_preds = [], []

        for valid_step, (x_valid, y_valid) in enumerate(valid_dataset):          
            y_valid_pred, valid_loss = self.valid_step(x_valid, y_valid)
            valid_epoch_loss += tf.reduce_mean(valid_loss)  # for each sample
            y_valid_preds.append(y_valid_pred)
            y_valid_trues.append(y_valid)

        valid_scores = []                
        if self.eval_metric:            
            y_valid_preds = tf.concat(y_valid_preds, axis=0)
            y_valid_trues = tf.concat(y_valid_trues, axis=0)
            
            for metric in self.eval_metric:                
                valid_scores.append(metric(y_valid_trues, y_valid_preds, transform=self.transform))                
        return valid_epoch_loss/(valid_step+1), valid_scores           

    def valid_step(self, x_valid, y_valid):
        # valid step for one batch
        y_valid_pred = self.model(x_valid, training=False)
        valid_loss = self.loss_fn(y_valid, y_valid_pred)   
        return y_valid_pred, valid_loss

    def predict(self, test_dataset):
        # predict the new data
        return

    def export_model(self, model_dir, only_pb=True):
        # save the model
        tf.saved_model.save(self.model, model_dir)
        print("protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            print("model weights successfully saved in {}.ckpt".format(model_dir))
