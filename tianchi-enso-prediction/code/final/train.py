import sys
sys.path.insert(0, './Time-series-prediction')
# tfts is an open-souced TensorFlow time series package: https://github.com/LongxingTan/Time-series-prediction
# welcome to star and contribute

import tensorflow as tf
from tfts import build_tfts_model, Trainer
from utils import set_seed, eval_score, eval_rmse, custome_rmse_fn
from dataset import prepare_2d_feature, prepare_2d_label, load_cmip, load_soda, DataLoaderTF
from tensorflow.keras.layers import Layer, Input, Embedding


def build_data(base_dir, batch_size):
    train = prepare_2d_feature(data_dir=base_dir + '/enso_round1_train_20210201/CMIP_train.nc')
    label = prepare_2d_label(label_dir=base_dir + '/enso_round1_train_20210201/CMIP_label.nc')

    SODA_train = prepare_2d_feature(data_dir=base_dir + '/enso_round1_train_20210201/SODA_train.nc')    
    SODA_label = prepare_2d_label(label_dir=base_dir + '/enso_round1_train_20210201/SODA_label.nc')

    train_data_reader = load_cmip(data=train, label=label, random_select=True, filter_1_99=True)
    valid_data_reader = load_soda(data=SODA_train, label=SODA_label, random_select=True)

    train_dataloader = DataLoaderTF(train_data_reader)(batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoaderTF(valid_data_reader)(batch_size=batch_size)
    return train_dataloader, valid_dataloader


class ENSO(object):
    # ENSO is a class to capture the feature for model
    def __init__(self):
        self.embed1 = Embedding(input_dim=12, output_dim=4)  # input_dim: vocab_size, output_dim: embed_size, input_length: input sequence length
        self.embed2 = Embedding(input_dim=4, output_dim=2)

    def forward(self, sst, t300, ua, va, month):
        """
        :param sst: batch * 12 * 24 * 72 
        :param t300: batch * 12 * 24 * 72 
        :param ua: batch * 12 * 24 * 72 
        :param va: batch * 12 * 24 * 72 
        '''
        lon: 72, NINO3.4: 190 ~ 240
        [  0.   5.  10.  15.  20.  25.  30.  35.  40.  45.  50.  55.  60.  65.
        70.  75.  80.  85.  90.  95. 100. 105. 110. 115. 120. 125. 130. 135.
        140. 145. 150. 155. 160. 165. 170. 175. 180. 185. 190. 195. 200. 205.
        210. 215. 220. 225. 230. 235. 240. 245. 250. 255. 260. 265. 270. 275.
        280. 285. 290. 295. 300. 305. 310. 315. 320. 325. 330. 335. 340. 345.
        350. 355.]
        lat: 24, NINO3.4: -5 ~ 5
        [-55. -50. -45. -40. -35. -30. -25. -20. -15. -10.  -5.   0.   5.  10.
        15.  20.  25.  30.  35.  40.  45.  50.  55.  60.]

        # NINO3.4区域, 190 ~ 240, 时序信息, 可以考虑时序的var、max、min、diff等统计信息, 以及月份等时间信息
        # 360 - (170–120°W) × 5°S–5°N
        # The NINO regions cover the following areas:
        # NINO1: 5-10°S, 80-90°W, 270~280
        # NINO2: 0-5°S, 80-90°W
        # NINO3: 5°N-5°S, 150-90°W, 210~280
        # NINO3.4: 5°N-5°S, 120-170°W
        # NINO4: 5°N-5°S, 160°E-150°W,  160~210
        # IOD west: 50°E to 70°E and 10°S to 10°N, 
        # IOD east: 90°E to 110°E and 10°S to 0°S
        """
        sst_nino12 = sst[..., 11: 14, 54 :57]
        sst_nino3 = sst[..., 10: 13, 42: 57]
        sst_nino34 = sst[..., 10: 13, 38: 49]  # batch * 12 * 3 * 11
        sst_nino4 = sst[..., 10:13, 32: 42]  

        sst_equator = sst[..., 10: 13, 32: 47]
        sst_equator = tf.reduce_mean(sst_equator, axis=2)  # batch * 12 * 15         

        sst_roi_mean = tf.expand_dims(tf.reduce_mean(sst_nino34, axis=(2, 3)), -1)
        sst_roi_max = tf.expand_dims(tf.reduce_max(sst_nino34, axis=(2, 3)), -1)
        sst_roi_var = tf.expand_dims(tf.math.reduce_variance(sst_nino34, axis=(2, 3)), -1)

        sst_lag2 = tf.roll(sst_roi_mean, shift=2, axis=1)
        sst_mean_lag2 = tf.concat([tf.zeros_like(sst_lag2)[:, :2], sst_lag2[:, 2:]], axis=1)

        sst_nino12_mean = tf.expand_dims(tf.reduce_mean(sst_nino12, axis=(2, 3)), -1)         
        sst_nino3_mean = tf.expand_dims(tf.reduce_mean(sst_nino3, axis=(2, 3)), -1)
        sst_nino4_mean = tf.expand_dims(tf.reduce_mean(sst_nino4, axis=(2, 3)), -1) 

        sst_43_mean_diff = sst_nino4_mean - sst_nino3_mean   
        tni = sst_nino4_mean - sst_nino12_mean   

        t300_roi = t300[..., 10: 13, 38: 49]  
        t300_roi_mean = tf.expand_dims(tf.reduce_mean(t300_roi, axis=(2, 3)), -1)  
        t300_lag2 = tf.roll(t300_roi_mean, shift=2, axis=1)
        t300_mean_lag2 = tf.concat([tf.zeros_like(t300_lag2)[:, :2], t300_lag2[:, 2:]], axis=1)

        encoder_month_embed = self.embed1(tf.cast(month, tf.int32))
        quarter = tf.cast(tf.divide(month, 3), tf.int32)
        encoder_quarter_embed = self.embed2(quarter)
        decoder_month_embed = tf.concat([encoder_month_embed, encoder_month_embed], axis=1)
        decoder_quarter_embed = tf.concat([encoder_quarter_embed, encoder_quarter_embed], axis=1)        

        encoder_month = tf.one_hot(tf.cast(month, tf.int32), depth=12)  # => batch * 12 * 12   
        decoder_month = tf.concat([encoder_month, encoder_month], axis=1)  # batch * 24
        decode_idx = tf.tile(tf.expand_dims(tf.range(24), 0), (tf.shape(decoder_month)[0], 1))
        decode_idx = tf.cast(tf.one_hot(decode_idx, depth=24), tf.float32) # batch * 12 * 12
           
        sst = sst_roi_mean
        encoder_inputs = tf.concat([sst_roi_max, sst_roi_var, sst_nino12_mean, sst_nino3_mean, sst_nino4_mean, sst_43_mean_diff, sst_equator, tni, t300_roi_mean, sst_mean_lag2, t300_mean_lag2, encoder_month_embed, encoder_quarter_embed], axis=-1)  # batch * 12 * 2
        decoder_inputs = tf.concat([decoder_month_embed, decoder_quarter_embed, decode_idx], axis=-1)
        return sst, encoder_inputs, decoder_inputs

    def __call__(self, x):
        sst, t300, ua, va, month = x
        output = self.forward(sst, t300, ua, va, month)
        return output


def build_model():
    model_params = {
        'predict_sequence_length': 24,
        'rnn_size': 62,
        'use_attention': False,
        'teacher_forcing': False,
    }

    inputs = (Input([12, 24, 72]), Input([12, 24, 72]), Input([12, 24, 72]), Input([12, 24, 72]), Input([12]))
    enso_inputs = ENSO()(x=inputs)
    outputs = build_tfts_model(use_model='seq2seq', custom_model_params=model_params, dynamic_decoding=True)(enso_inputs)
    model = tf.keras.Model(inputs, outputs)
    return model


def run_train():
    fit_params = {
        'n_epochs': 22,
        'batch_size': 32,
        'learning_rate': 1.6e-4,
        'eval_metric': [eval_score, eval_rmse],
        'stop_no_improve_epochs': None, 
        'model_dir': '../user_data/nn'
    }

    train_dataloader, valid_dataloader = build_data(base_dir='../tcdata', batch_size=fit_params['batch_size'])  # prepare the data
    model = build_model()  # prepare the model
    trainer = Trainer(model=model, loss=custome_rmse_fn, optimizer=tf.keras.optimizers.Adam())  # load it to trainer
    trainer.train(train_dataloader, valid_dataloader, **fit_params)  # start to train


if __name__ == '__main__':
    set_seed(315)
    run_train()
