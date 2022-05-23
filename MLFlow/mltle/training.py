import tensorflow as tf
import numpy as np
from keras import backend as K


class LossWithMemoryCallback(tf.keras.callbacks.Callback):
    """
    This is loss helper.
    This is a callback that remembers the last batch error for each target variable 
    and sets its discounted version as the default filling value for missing values errors
    
    Parameters
    ----------
    variables: Dict[Str, keras.variable]
        dictionary of variables for each target variable
        
    discount: Float, default=0.6
        which fraction of error to propagate:
        error = last_error*discount
    
    """
    def __init__(self, variables, discount=0.6, decay = 0.8):
        for k, v in variables.items():
            setattr(self, k, v)

        self.vars = variables.keys()
        self.discount = discount
        self.decay = decay

    def on_batch_end(self, batch, logs):
        for k in self.vars:
            var = getattr(self, k)
            K.set_value(var, logs[f'{k}_loss']**0.5 * self.discount)

    def on_epoch_end(self, epoch, logs=None):
        self.discount = self.discount*self.decay


def mse_loss_wrapper(var):
    """
    This is loss wrapper.
    Passes keras.variable that is changed on each batch for loss calculation
    """
    def mse_nan_with_memory(y_true, y_pred):
        """
        This is loss. 
        Fills missing error with last known error
        """
        masked_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        masked_pred = tf.where(tf.math.is_nan(y_true), var, y_pred)
        error = K.mean(K.square(masked_pred - masked_true), axis=-1)
        return error
    return mse_nan_with_memory


class Model:
    """
    This is multi-task drug-target binding strength prediction model:
    MuLti-Task with Label Encoding.
    

    Parameters
    ----------
    drug_emb_size : Int, default=64
        embedding vector size for drug

    protein_emb_size: Int, default=32
        embedding vector size for protein
    
    max_drug_len: Int, default=200
        maximum length of the drug sequence to be passed. This is necessary for positional encoding

    drug_aphabet_len: Int, default=53
        number of unique symbols in all drug sequences,
        here default alphabet is unique characters of SMILES string
    
    protein_alphabet_len: Int, default=8006
        number of unique symbols in all protein sequences,
        here default alphabet is unique triplets of protein string, 
        number is obtained by calculating all 3-mers from BindingDB target human proteins
        
    
    Author
    ----------
    Elisaveta V
    """
    def __init__(self,
                 drug_emb_size=64,
                 protein_emb_size=32,
                 max_drug_len=200,
                 drug_alphabet_len=53,
                 protein_alphabet_len=8006):

        self.drug_emb_size = drug_emb_size
        self.protein_emb_size = protein_emb_size
        self.max_drug_len = max_drug_len
        self.drug_alphabet_len = drug_alphabet_len
        self.protein_alphabet_len = protein_alphabet_len

    def _connv_batchnorm(self,
                         inp,
                         filters,
                         kernel,
                         initializer,
                         regularizer,
                         activation):

        cnv = tf.keras.layers.Conv1D(filters,
                     kernel,
                     padding='same',
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizer,
                     activation=None)(inp)
        cnv = tf.keras.layers.BatchNormalization()(cnv)
        if activation == 'relu':
            cnv = tf.keras.layers.ReLU()(cnv)

        return cnv

    def create_model(self,
              order=[],
              activations={},
              activation = 'relu',
              pooling_mode = 'max',
              num_res_blocks=3,
              units_per_head=32,
              units_per_layer=128,
              dropout_rate=0.3,
              drug_kernel=(2, 3),
              protein_kernel=(7, 7),
              loss_weights = [],
              usemetrics =  {},
              uselosses = {},
              initializer=None,
              regularizer=None,
              optimizer = None,
              drug_strides_up=1,
              protein_strides_down=1,
              run_eagerly=True, 
              positional=True):
        """
        Returns a compiled tensorflow model according to the passed parameters
        
        
        Parameters
        ----------
        
        order: List[Str]
            names of the target variables in the order in which they will be passed to the model.
            It is used to create the head layers and apply certain activation functions
        
        units_per_head: Int, default=32
            dimensionality of the output space in head layers
        
        units_per_layer: Int, default=128
            dimensionality of the output space in shared representation layer
        
        dropout_rate: Float, default=0.3
        
        num_res_blocks, Int, default=3
            number of residual blocks to create
        
        activations: Dict[Str, Str]
            dictionary that matches the names of target variables from the order list 
            with the corresponding activation function used in the output layer
        
        initializer: Str or tensorflow.keras.initializers object
            tensorflow.keras.initializers compatible kernel initializer
        
        drug_kernel: Tuple(Int, Int)
            kernel size for the drug convolutional layer for the 1st and 2nd layers in each residual block
            
        protein_kernel: Tuple(Int, Int)
            kernel size for the protein convolutional layer for the 1st and 2nd layers in each residual block
            
            
        
        Returns
        ----------
        model: tensorflow.keras.models.Model
            compiled tensorflow.keras.models model
        """

        drug_emb_size = self.drug_emb_size
        protein_emb_size = self.protein_emb_size
        max_drug_len = self.max_drug_len + 1
        drug_alphabet_len = self.drug_alphabet_len
        protein_alphabet_len = self.protein_alphabet_len

        num_filters_drug_1 = drug_emb_size // 2
        num_filters_drug_2 = drug_emb_size

        num_filters_protein_1 = protein_emb_size // 2
        num_filters_protein_2 = protein_emb_size

        drug_seq = tf.keras.layers.Input(shape=(None, ), name='drug_seq_inp')
        protein_seq = tf.keras.layers.Input(shape=(None, ), name='protein_seq_inp')
        drug_pos = tf.keras.layers.Input(shape=(None, ), name='drug_pos_inp')

        drug_seq_emb = tf.keras.layers.Embedding(drug_alphabet_len,
                                 drug_emb_size,
                                 trainable=True,
                                 name='drug_seq_emb',
                                 mask_zero=True,
                                 embeddings_initializer=initializer)(drug_seq)

        if positional:

          drug_seq_emb = tf.keras.layers.Embedding(drug_alphabet_len,
                         drug_emb_size,
                         trainable=True,
                         name='drug_seq_emb',
                         mask_zero=True,
                         embeddings_initializer=initializer)(drug_seq)

          drug_pos_emb = tf.keras.layers.Embedding(max_drug_len,
                                   drug_emb_size,
                                   trainable=True,
                                   name='drug_pos_emb',
                                   embeddings_initializer=initializer)(drug_pos)

          drug_emb = tf.keras.layers.Add(name='drug_emb')([drug_seq_emb, drug_pos_emb])

        else:

          drug_emb = tf.keras.layers.Embedding(drug_alphabet_len,
                         drug_emb_size,
                         trainable=True,
                         name='drug_seq_emb',
                         mask_zero=True,
                         embeddings_initializer=initializer)(drug_seq)


        protein_emb = tf.keras.layers.Embedding(
            protein_alphabet_len,
            protein_emb_size,
            trainable=True,
            name='protein_emb',
            mask_zero=True,
            embeddings_initializer=initializer)(protein_seq)

        drug_skip_con = tf.keras.layers.Conv1DTranspose(drug_emb_size,
                               drug_kernel[0],
                               strides = drug_strides_up,
                               name='drug_skip_con_1',
                               kernel_initializer=initializer,
                               kernel_regularizer=regularizer,
                               activation=activation)(drug_emb)

        protein_skip_con = tf.keras.layers.Conv1D(protein_emb_size,
                                  protein_kernel[1],
                                  strides = protein_strides_down,
                                  name='protein_skip_con_1',
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  activation=activation)(protein_emb)

        for b in range(2, num_res_blocks + 2):

            drug_cnv = self._connv_batchnorm(drug_skip_con, num_filters_drug_1,
                                        drug_kernel[0], initializer,regularizer, activation)
            drug_cnv = self._connv_batchnorm(drug_cnv, num_filters_drug_2,
                                        drug_kernel[1], initializer,regularizer, None)

            protein_cnv = self._connv_batchnorm(protein_skip_con,
                                           num_filters_protein_1,
                                           protein_kernel[0], initializer, regularizer,
                                           activation)
            protein_cnv = self._connv_batchnorm(protein_cnv, num_filters_protein_2,
                                           protein_kernel[1], initializer, regularizer, None)

            drug_skip_con = tf.keras.layers.Add(name=f'drug_skip_con_{b}')([drug_skip_con, drug_cnv])
            protein_skip_con = tf.keras.layers.Add(name=f'protein_skip_con_{b}')([protein_skip_con, protein_cnv])

            if activation == 'relu':
                drug_skip_con = tf.keras.layers.ReLU()(drug_skip_con)
                protein_skip_con = tf.keras.layers.ReLU()(protein_skip_con)

        if pooling_mode == 'max':
            drug_global_pooled = tf.keras.layers.GlobalMaxPool1D(name='drug_global_pooled')(drug_skip_con)
            protein_global_pooled = tf.keras.layers.GlobalMaxPool1D(name='protein_global_pooled')(protein_skip_con)
        elif pooling_mode == 'avg':
            drug_global_pooled = tf.keras.layers.GlobalAveragePooling1D(name='drug_global_pooled')(drug_skip_con)
            protein_global_pooled = tf.keras.layers.GlobalAveragePooling1D(name='protein_global_pooled')(protein_skip_con)

        drug_protein_vec = tf.keras.layers.concatenate([drug_global_pooled, protein_global_pooled], axis=-1, name='drug_protein_vec')
        #drug_protein_vec = tf.keras.layers.BatchNormalization()(drug_protein_vec)
        #drug_protein_vec = tf.keras.layers.Dropout(dropout_rate)(drug_protein_vec)

        shared_layer = tf.keras.layers.Dense(units_per_layer, name='shared_layer_1', activation=activation)(drug_protein_vec)
        #shared_layer = tf.keras.layers.BatchNormalization()(shared_layer)
        shared_layer = tf.keras.layers.Dropout(dropout_rate)(shared_layer)


        shared_layer = tf.keras.layers.Dense(units_per_layer, name='shared_layer_2', activation=activation)(shared_layer)
        #shared_layer = tf.keras.layers.BatchNormalization()(shared_layer)
        shared_layer = tf.keras.layers.Dropout(dropout_rate)(shared_layer)

        head_layers = {}
        for out in order:
            head = tf.keras.layers.Dense(units_per_head,
                                         activation=activation,
                                         name=f'head_{out}',
                                         kernel_initializer=initializer)(shared_layer)
            # head = tf.keras.layers.BatchNormalization()(head)
            head_layers[out] = head

        outputs = []
        for out in order:
            head = head_layers[out]
            output = tf.keras.layers.Dense(1,
                           activation=activations[out],
                           name=out,
                           kernel_initializer=initializer)(head)
            outputs.append(output)

        model = tf.keras.models.Model(inputs=[drug_seq, protein_seq, drug_pos],
                             outputs=outputs)

        model.compile(loss=uselosses,
                      optimizer=optimizer,
                      metrics=usemetrics,
                      loss_weights=loss_weights,
                      run_eagerly=run_eagerly)

        print(f"Done. Total params: {model.count_params()}")
        return model