import tensorflow as tf
import random
import pandas as pd
import numpy as np
import os
from keras import backend as K
from tqdm.auto import tqdm

preprocessing = tf.keras.preprocessing

CHARPROTSET = dict([('A', 1), ('G', 2), ('L', 3), ('M', 4), ('S', 5), ('T', 6),
                    ('E', 7), ('Q', 8), ('P', 9), ('F', 10), ('R', 11),
                    ('V', 12), ('D', 13), ('I', 14), ('N', 15), ('Y', 16),
                    ('H', 17), ('C', 18), ('K', 19), ('W', 20), ('X', 21)])

CHARCANSMISET = dict([(')', 1), ('(', 2), ('1', 3), ('C', 4), ('c', 5),
                      ('O', 6), ('2', 7), ('N', 8), ('=', 9), ('n', 10),
                      ('3', 11), ('-', 12), ('4', 13), ('F', 14), ('S', 15),
                      ('[', 16), (']', 17), ('l', 18), ('H', 19), ('s', 20),
                      ('#', 21), ('o', 22), ('5', 23), ('B', 24), ('r', 25),
                      ('+', 26), ('6', 27), ('P', 28), ('.', 29), ('I', 30),
                      ('7', 31), ('e', 32), ('i', 33), ('a', 34), ('8', 35),
                      ('K', 36), ('A', 37), ('9', 38), ('T', 39), ('g', 40),
                      ('R', 41), ('Z', 42), ('%', 43), ('0', 44), ('u', 45),
                      ('V', 46), ('b', 47), ('t', 48), ('L', 49), ('*', 50),
                      ('d', 51), ('W', 52)])


def mse_nan(y_true, y_pred):

    masked_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
    is_empty = tf.equal(tf.size(masked_true), 0)

    if is_empty:
        error = tf.constant(0.0)
        return error

    error = K.mean(K.square(masked_pred - masked_true), axis=-1)
    return error


class LossWithMemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, kd_var, ki_var, ic_var, ec_var, discount=0.01):
        self.kd_var = kd_var
        self.ki_var = ki_var
        self.ic_var = ic_var
        self.ec_var = ec_var

        self.discount = discount

    def on_epoch_end(self, epoch, logs):
        self.kd_var = logs['Kd_loss']*self.discount
        self.ki_var = logs['Ki_loss']*self.discount
        self.ic_var = logs['IC50_loss']*self.discount
        self.ec_var = logs['EC50_loss']*self.discount


def mse_nan_with_memory(y_true, y_pred, var):

    masked_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred)
    is_empty = tf.equal(tf.size(masked_true), 0)

    if is_empty:
        error = var
        return error

    error = K.mean(K.square(masked_pred - masked_true), axis=-1)
    return error


class Gen:
    def __init__(self,
                 data,
                 map_smiles,
                 map_aa,
                 shuffle=True,
                 test_only=False,
                 len_drug=100,
                 len_target=1000,
                 window=False):
        self.data = data
        self.map_smiles = map_smiles
        self.map_aa = map_aa
        self.shuffle = shuffle
        self.test_only = test_only
        self.len_drug = len_drug
        self.len_target = len_target
        self.size = self.data.shape[0]
        self.inds = list(range(self.size))
        if self.shuffle:
            random.shuffle(self.inds)

        self.window = window

        self.gen = self._get_inputs()

    def _get_inputs(self):
        seen = 0
        while seen < self.size:
            ind = self.inds[seen]
            sample = self.data.iloc[ind, :].values.tolist()
            sample[0] = self.map_smiles[sample[0]]
            sample[1] = self.map_aa[sample[1]]

            if self.window:
                ld = max(0, (len(sample[0]) - self.len_drug))
                lt = max(0, (len(sample[1]) - self.len_target))
                dstart = random.randint(0, ld)
                tstart = random.randint(0, lt)

                sample[0] = sample[0][dstart:dstart + self.len_drug]
                sample[1] = sample[1][tstart:dstart + self.len_target]

            yield sample
            seen += 1
            if seen == self.size:
                if self.shuffle:
                    random.shuffle(self.inds)
                seen = 0

    def get_batch(self, batch_size):
        while True:
            BATCH = []
            for _ in range(batch_size):
                sample = next(self.gen)
                for k, value in enumerate(sample):
                    if len(BATCH) < (k+1):
                        BATCH.append([])
                    BATCH[k].append(value)

            BATCH[0] = preprocessing.sequence.pad_sequences(BATCH[0], self.len_drug)
            BATCH[1] = preprocessing.sequence.pad_sequences(BATCH[1], self.len_target)

            for k in range(2, len(BATCH)):
                BATCH[k] = np.array(BATCH[k]).flatten()

            if not self.test_only:
                yield [BATCH[0],  BATCH[1]], [BATCH[k] for k in range(2, len(BATCH))]
            else:
                yield [BATCH[0],  BATCH[1]], [BATCH[k]*0 for k in range(2, len(BATCH))]


class MLTLE:
    def __init__(self, mse_nan=mse_nan, mse_nan_with_memory=mse_nan_with_memory):
        self.model = model = tf.keras.models.load_model(
            'data/mltle/cnn_with_memory.hdf5')

        self.target_columns = ['p1Kd', 'p1Ki',
                               'p1IC50', 'p1EC50', 'is_active', 'pH']

    def predict(self, data):
        for col in self.target_columns:
            data[col] = 0

        SMILES = {}
        for smiles in tqdm(data['smiles'].unique()):
            SMILES[smiles] = [CHARCANSMISET[s] for s in smiles]

        AA = {}
        for aa in tqdm(data['target'].unique()):
            AA[aa] = [CHARPROTSET[a.upper()] for a in aa]

        teg = Gen(data, SMILES, AA, shuffle=False, test_only=True)
        teg = teg.get_batch(1)

        prediction = self.model.predict(teg, steps=data.shape[0], verbose=1)

        for k, col in enumerate(self.target_columns):
            data[col] = prediction[k].ravel()

        return data

    def to_nM(self, x):
        return np.expm1(x)

    def to_uM(self, x):
        return np.expm1(x)/1000

    def to_pKd(self, x):
        return -np.log10(np.expm1(x) / 10**9)
