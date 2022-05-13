import tensorflow as tf
import numpy as np
import os

from mltle.datamap import MapSeq
from mltle.datagen import DataGen 
from mltle.data import spec


class Model:
    """

    Prediction interface
    
    Parameters
    ----------
    model_type: Str, default='Res2_06'
        model name, available : 'Res2_06' 
        
    Example
    ----------
    model = mlt.predict.Model("Res2_06")
    predictions = model.predict(X_test)
    """
    def __init__(self, model_type, model_spec=None):
        self.model_path = os.path.join(os.path.dirname(__file__), f'data/models/{model_type}.hdf5')
        self.model = tf.keras.models.load_model(self.model_path)

        if not model_spec:
            self.model_spec = getattr(spec, model_type)
        else:
            self.model_spec = model_spec

        self.drug_mode  = self.model_spec['drug_mode']
        self.protein_mode = self.model_spec['protein_mode']
        self.max_drug_len = self.model_spec['max_drug_len']
        self.output_order = self.model_spec['output_order']
        self.scalers =  self.model_spec['scalers']

    def get_batch_size(self, data_size, max_batch_size = 128):
        max_data_batch_size = 1

        for i in range(1, min(max_batch_size, data_size)):
            if data_size % i == 0:
                max_data_batch_size = i

        assert data_size % max_data_batch_size == 0

        return max_data_batch_size

    def inverse_scale(self, x, output):
        mean, std = self.scalers[output]
        x*=std
        x+=mean
        return x


    def _get_maps(self, drug_seqs, protein_seqs):
        mapseq = MapSeq(drug_mode=self.drug_mode, protein_mode=self.protein_mode, max_drug_len=self.max_drug_len)
        map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)
        return map_drug, map_protein

    def predict(self, data):
        """

        Parameters
        ----------
        data: pandas.DataFrame
            Expected data should have two columns:
            ['drug', 'target']

        Returns
        ----------
        data: pandas.DataFrame
            columns order:
            [drug_sequence, protein_sequence, target_1, target_2, ... target_n]
            [    Str,              Str,        Float,    Float,   ...  Float]

        """
        drug_seqs, protein_seqs = data.iloc[:, 0].unique(), data.iloc[:, 1].unique()
        map_drug, map_protein = self._get_maps(drug_seqs, protein_seqs)

        max_data_batch_size = self.get_batch_size(data.shape[0])
        print(f'`data_size`={data.shape[0]}, `max_data_batch_size`={max_data_batch_size}')

        data_gen = DataGen(data, map_drug, map_protein, shuffle=False, test_only=True)
        data_gen = data_gen.get_generator(max_data_batch_size)

        print('Predicting ...')
        predictions = self.model.predict(data_gen, steps=data.shape[0] // max_data_batch_size , verbose=1)

        for k, col in enumerate(self.output_order):
            prediction = predictions[k].ravel()

            if col in self.scalers.keys():
                prediction = self.inverse_scale(prediction, col)
            data[col] = prediction


        return data


    def to_nM(self, x):
        return np.expm1(x)

    def to_uM(self, x):
        return np.expm1(x)/1000

    def to_pKd(self, x):
        return -np.log10(np.expm1(x) / 10**9)
