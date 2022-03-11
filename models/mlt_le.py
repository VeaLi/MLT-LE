import tensorflow as tf
import random
import pandas as pd
import numpy as np
import os
from keras import backend as K
from tqdm.auto import tqdm

preprocessing = tf.keras.preprocessing

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}


CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                 "t": 61, "y": 62, "*": 63}


def mse_nan(y_true, y_pred):
    masked_true = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_true)
    masked_pred = tf.where(tf.math.is_nan(
        y_true), tf.zeros_like(y_true), y_pred)
    return K.mean(K.square(masked_pred - masked_true), axis=-1)


bl = tf.keras.losses.BinaryCrossentropy()


def loss(y_true, y_pred):
    return bl(y_true, y_pred)


class Gen:
    def __init__(self, data, map_smiles, map_aa, shuffle=True, test_only=False):
        self.data = data
        self.map_smiles = map_smiles
        self.map_aa = map_aa
        self.shuffle = shuffle
        self.test_only = test_only
        self.size = self.data.shape[0]
        self.inds = list(range(self.size))
        if self.shuffle:
            random.shuffle(self.inds)

        self.gen = self._get_inputs()

    def _get_inputs(self):
        seen = 0
        while seen < self.size:
            ind = self.inds[seen]
            drug, target, kd, ki, ic50, ec50, ia = self.data.iloc[ind, :]
            drug = self.map_smiles[drug]
            target = self.map_aa[target]
            drug = preprocessing.sequence.pad_sequences([drug], 100)[0]
            target = preprocessing.sequence.pad_sequences([target], 1000)[0]

            yield drug.tolist(), target.tolist(), kd, ki, ic50, ec50, ia
            seen += 1
            if seen == self.size:
                if self.shuffle:
                    random.shuffle(self.inds)
                seen = 0

    def get_batch(self, batch_size):
        while True:
            D, T, K, I, C, E, IA = [], [], [], [], [], [], []
            for _ in range(batch_size):
                d, t, k, i, c, e, ia = next(self.gen)
                D.append(d)
                T.append(t)
                K.append(k)
                I.append(i)
                C.append(c)
                E.append(e)
                IA.append(ia)

            D = np.array(D)
            T = np.array(T)
            K = np.array(K)
            I = np.array(I)
            C = np.array(C)
            E = np.array(E)
            IA = np.array(IA)

            if not self.test_only:
                yield [D, T], [K.flatten(), I.flatten(), C.flatten(),
                               E.flatten(), IA.flatten()]
            else:
                yield [D, T], [K.flatten() * 0, I.flatten() * 0, C.flatten() * 0,
                               E.flatten() * 0, IA.flatten() * 0]


class MLTLE:
    def __init__(self, mse_nan=mse_nan, loss=loss):
        # NB. it is not focal loss, it wrong alias
        self.model = model = tf.keras.models.load_model(
            'data/mltle/06-4.07.hdf5', custom_objects={'mse_nan': mse_nan, 'focal_loss': loss})

        self.target_columns = ['Ki (nM) log1p',
                               'IC50 (nM) log1p',
                               'Kd (nM) log1p',
                               'EC50 (nM) log1p',
                               'is_active']

    def predict(self, data):
        for col in self.target_columns:
            data[col] = 0

        SMILES = {}
        for smiles in tqdm(data['smiles'].unique()):
            SMILES[smiles] = [CHARCANSMISET[s] for s in smiles][:100]

        AA = {}
        for aa in tqdm(data['target'].unique()):
            AA[aa] = [CHARPROTSET[a.upper()] for a in aa][:1000]

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
