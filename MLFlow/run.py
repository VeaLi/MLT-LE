import sys
sys.path.insert(0, '../')


import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow import log_metrics

import argparse

from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback


from keras import backend as K
import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os


import mltle as mlt



  



class MlFlowCallback(tf.keras.callbacks.Callback):
    """
    MLFlow Callback adapted from:
        https://github.com/gnovack/celeb-cnn-project
    """

    def on_epoch_end(self, epoch, logs=None):
        log_metrics(logs, step=epoch)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def run_experiment(run, experiment_name, num_epochs, batch_size, num_res_blocks, discount, data, use_loss_weights):


    data = pd.read_csv(data)
    order = data.columns[2:]

    if use_loss_weights:
        loss_weights = softmax(1 - data.iloc[:,2:].notna().mean()).values.tolist()
    else:
        loss_weights = [1.0] * len(order)



    model = mlt.training.Model(drug_emb_size=128,
                              protein_emb_size=64,
                              max_drug_len=200,
                              drug_alphabet_len=53,
                              protein_alphabet_len=8006)

    
    

    variables = {}
    for var in order:
        variables[var] = K.variable(0.0)

    LossCallback = mlt.training.LossWithMemoryCallback(variables, discount=discount, decay = 0.8)

    uselosses = defaultdict(lambda: mlt.training.mse_loss_wrapper)
    uselosses['is_active'] = 'binary_crossentropy'

    for k, v in variables.items():
        if k not in uselosses.keys():
            uselosses[k] = uselosses[k](v)

    usemetrics = {'is_active': tf.keras.metrics.AUC()}

    activations = defaultdict(lambda: 'linear')
    activations['is_active'] = 'sigmoid'


    initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='normal', seed=7)
    optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.Nadam(), sync_period=3)

    model = model.create_model(order=order,
                                activations=activations,
                                activation = 'relu',
                                pooling_mode = 'max',
                                num_res_blocks=num_res_blocks,
                                units_per_head=64,
                                units_per_layer=256,
                                dropout_rate=0.3,
                                drug_kernel=(2, 3),
                                protein_kernel=(7, 7),
                                loss_weights=loss_weights,
                                usemetrics=usemetrics,
                                uselosses=uselosses,
                                initializer=initializer,
                                optimizer=optimizer,
                                drug_strides_up=1,
                                protein_strides_down=1)


    mapseq = mlt.datamap.MapSeq(drug_mode='smiles_1', protein_mode='protein_3', max_drug_len=200)
    drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()
    map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)


    X_train, X_valid = train_test_split(data, test_size=0.3, shuffle=True, random_state=7, stratify=data['is_active'])


    train_gen = mlt.datagen.DataGen(X_train, map_drug, map_protein)
    train_gen = train_gen.get_generator(batch_size)

    valid_gen = mlt.datagen.DataGen(X_valid, map_drug, map_protein)
    valid_gen = valid_gen.get_generator(batch_size)

    steps_per_epoch = X_train.shape[0] // batch_size
    valid_steps = X_valid.shape[0] // batch_size

    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    CSVLoggerCallback = tf.keras.callbacks.CSVLogger(f"logs/{experiment_name}.log")
    history = model.fit(train_gen,
                    validation_data=valid_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=valid_steps,
                    verbose=0,
                    callbacks=[TqdmCallback(), MlFlowCallback(), LossCallback, CSVLoggerCallback],
                    epochs=num_epochs)


def main(experiment_name, epochs, batch_size, num_res_blocks, discount, data, use_loss_weights):
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    print("Logging with mlflow.tensorflow.autolog")
    mlflow.tensorflow.autolog()

    with mlflow.start_run() as run:
        print("MLflow:")
        print("run_id:", run.info.run_id)
        print("experiment_id:", run.info.experiment_id)

        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("mlflow_tensorflow.autolog", True)

        run_experiment(run, experiment_name, epochs, batch_size, num_res_blocks, discount, data, use_loss_weights)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name')
    parser.add_argument('--batch_size')
    parser.add_argument('--epochs')
    parser.add_argument('--num_res_blocks')
    parser.add_argument('--discount')
    parser.add_argument('--data')
    parser.add_argument('--use_loss_weights')
    args = parser.parse_args()

    experiment_name = str(args.experiment_name)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    num_res_blocks = int(args.num_res_blocks)
    discount = float(args.discount)
    data = str(args.data)
    use_loss_weights = bool(args.use_loss_weights)

    main(experiment_name, epochs, batch_size, num_res_blocks, discount, data, use_loss_weights)
