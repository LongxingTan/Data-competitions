# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01
# paper:
# other implementations:
#   https://github.com/Arturus/kaggle-web-traffic
#   https://github.com/pytorch/fairseq
#   https://github.com/LenzDu/Kaggle-Competition-Favorita/blob/master/seq2seq.py
#   https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Intro.ipynb
# Enhancement:
#   Residual LSTM:Design of a Deep Recurrent Architecture for Distant Speech... https://arxiv.org/abs/1701.03360
#   A Dual-Stage Attention-Based recurrent neural network for time series prediction. https://arxiv.org/abs/1704.02971

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRUCell, LSTMCell, RNN, GRU
from ..layers.attention_layer import Attention


params = {
    'rnn_size': 64,
    'dense_size': 16,
    'num_stacked_layers': 1,
    'use_attention': True,
    'teacher_forcing': True,
    'scheduler_sampling': False
}


class Seq2Seq(object):
    def __init__(self, custom_model_params, dynamic_decoding=True):
        params.update(custom_model_params)
        self.encoder = Encoder(params)
        self.decoder = Decoder2(params)
        self.params = params

    def __call__(self, inputs, teacher=None):
        if isinstance(inputs, tuple):
            x, encoder_feature, decoder_feature = inputs
            print(x.shape, encoder_feature.shape, decoder_feature.shape)
            encoder_feature = tf.concat([x, encoder_feature], axis=-1)
        else:  # for single variable prediction
            encoder_feature = x = inputs
            decoder_feature = None

        encoder_output, encoder_state = self.encoder(encoder_feature)

        decoder_init_input = x[:, -1, 0:1]
        init_state = encoder_state
        decoder_output = self.decoder(decoder_feature, init_state, decoder_init_input,
                                      encoder_output=encoder_output,
                                      teacher=teacher,
                                      use_attention=self.params['use_attention'])
        return decoder_output


class Encoder(object):
    def __init__(self, params):
        self.params = params
        self.rnn1 = GRU(units=64, activation='tanh', return_sequences=True, return_state=False, dropout=0.24)
        self.rnn2 = GRU(units=64, activation='tanh', return_sequences=True, return_state=True, dropout=0.24)
        self.dense1 = Dense(units=62, activation='tanh')

    def __call__(self, inputs, training=None, mask=None):
        # outputs: batch_size * input_seq_length * rnn_size, state: batch_size * rnn_size
        x = self.rnn1(inputs)
        encoder_output, encoder_state = self.rnn2(x)
        #encoder_hidden_state = tuple(self.dense(hidden_state) for _ in range(params['num_stacked_layers']))
        encoder_state = self.dense1(encoder_state)  # => batch_size * input_seq_length * dense_size
        return encoder_output, encoder_state


class Decoder2(object):
    def __init__(self, params):
        self.params = params
        self.predict_window_sizes = params['predict_sequence_length']
        self.rnn_cell = GRUCell(self.params['rnn_size'])
        self.dense = Dense(units=1)
        self.attention = Attention(hidden_size=32, num_heads=2, attention_dropout=0.8)

    def __call__(self, decoder_inputs, init_state, init_value, encoder_output, teacher=None, use_attention=False):
        decoder_outputs = []
        prev_output = init_value
        prev_state = init_state

        for i in range(self.predict_window_sizes):
            if teacher is None:
                this_input = tf.concat([prev_output, decoder_inputs[:, i]], axis=-1)
            else:
                this_input = tf.concat([teacher[:, i: i + 1], decoder_inputs[:, i]])
            if use_attention:
                att = self.attention((prev_state, encoder_output, encoder_output))
                this_input = tf.concat([this_input, att], axis=-1)

            this_output, this_state = self.rnn_cell(this_input, prev_state)
            prev_state = this_state
            prev_output = self.dense(this_output)
            decoder_outputs.append(prev_output)

        decoder_outputs = tf.concat(decoder_outputs, axis=-1)
        return decoder_outputs
