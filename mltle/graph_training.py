import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.utils.layer_utils import count_params


class GraphModel:
    """
    This is a multi-task drug-target binding strength prediction model:
    MuLti-Task with Label Encoding and Graph Encoding.


    Parameters
    ----------
    protein_emb_size: Int, default=64
        Embedding depth for tf.keras.layers.Embedding

    protein_alphabet_len: Int, default=8006
        Alphabet size for tf.keras.layers.Embedding
        If you use `protein_3` encoding the alphabet length is 8006,
        for the length of other encoding please see `mlt.data.spec` 



    Author
    ----------
    Elisaveta V
    """

    def __init__(self,
                 protein_emb_size=64,
                 protein_alphabet_len=8006):

        self.protein_emb_size = protein_emb_size
        self.protein_alphabet_len = protein_alphabet_len

    def _connv_batchnorm(self,
                         inp,
                         filters,
                         kernel,
                         initializer,
                         regularizer,
                         activation):
        """
        Order:
        Convolution -> Normalization -> Activation
        """

        cnv = tf.keras.layers.Conv1D(filters,
                                     kernel,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     activation=None)(inp)

        cnv = tf.keras.layers.BatchNormalization()(cnv)

        if activation == 'relu':
            cnv = tf.keras.layers.ReLU()(cnv)
        elif activation == 'selu':
            cnv = tf.keras.activations.selu()(cnv)

        return cnv

    def create_model(self,
                     order=[],
                     activations={},
                     activation='relu',
                     pooling_mode='max',
                     num_res_blocks=3,
                     units_per_head=32,
                     units_per_layer=128,
                     dropout_rate=0.3,
                     protein_kernel=(7, 7),
                     loss_weights=None,
                     usemetrics={},
                     uselosses={},
                     initializer=None,
                     regularizer=None,
                     optimizer=None,
                     protein_strides_down=1,
                     run_eagerly=True,
                     num_graph_features=78,
                     graph_depth=4,
                     graph_type='gin_eps0'):
        """
        Returns a compiled tensorflow model according to the passed parameters


        Parameters
        ----------

        order: List[Str]
            names of the target variables in the order in which they will be passed to the model.
            It is used to create the head layers and apply certain activation functions

        activations: Dict[Str, Str]
            dictionary that matches the names of target variables from the order list 
            with the corresponding activation function used in the output layer

        activation: Str, default='relu'
            'relu': use ReLu activation
            'selu': use SeLu for additional normalization in residual networks
            Note: inside grapg layers ReLu function is implemented

        pooling_mode: Str, default='max'
            "max": Implements Global Max Pooling
            "avg": Implements Global Average Pooling
            TODO: add description for "avg" case use 

        num_res_blocks, Int, default=3
            number of residual blocks to create

        units_per_head: Int, default=32
            dimensionality of the output space in head layers

        units_per_layer: Int, default=128
            dimensionality of the output space in shared representation layer

        dropout_rate: Float, default=0.3

        protein_kernel: Tuple(Int, Int)
            kernel size for the protein convolutional layer for the 1st and 2nd layers in each residual block

        loss_weights: List[Float]
            Represents weights for each output during loss calculation
            Dim(loss_weights) =  Dim(Order)

        usemetrics: Dict{Str:List[Str]}, default={}
            For additional metric calculation pass non empty dictionary.
            Example: {'pKd': ['mse', tf.keras.metrics.mean_absolute_error()]}

        uselosses: Dict{Str:Str}
            Specify loss type for each output.
            Example: {'pKd': 'mse', 'pIC50': 'mae'}

        initializer: Str or tensorflow.keras.initializers object
            tensorflow.keras.initializers compatible kernel initializer

        regularizer: Str or tensorflow.keras.regularizers object
            tensorflow.keras.regularizers compatible kernel regularizer

        optimizer: tf.keras.optimizers compatible object

        protein_strides_down: Int, default=1
            Use to reduce protein seq dimensionality in Conv1D

        run_eagerly: Bool, default=True
            Train model in eager mode, allows operations on Graphs durign training, such
            as tensor.numpy(), maybe required for LossWithMemory calculation

        num_graph_features: Int, default=78
            Number of features in chosen graph model
            For example model `G78` has 78 features

        graph_depth: Int, default=4
            number of graph layers in drug encoding

        graph_type: Str, default='gin_eps0'
            "gin_eps0": Graph Isomorphism Network with eps=0
            "gcn": Graph Convolution Neural Network



        Returns
        ----------
        model: tensorflow.keras.models.Model
            compiled tensorflow.keras.models model
        """

        protein_emb_size = self.protein_emb_size
        protein_alphabet_len = self.protein_alphabet_len

        num_filters_protein_1 = protein_emb_size // 2
        num_filters_protein_2 = protein_emb_size

        input_nodes = tf.keras.layers.Input(
            shape=(None, num_graph_features), name='nodes', dtype='float32')
        input_adj = tf.keras.layers.Input(
            shape=(None, None), name='adj', dtype='float32')

        protein_seq = tf.keras.layers.Input(
            shape=(None, ), name='protein_seq_inp')
        drug_pos = tf.keras.layers.Input(shape=(None, ), name='drug_pos_inp')

        protein_emb = tf.keras.layers.Embedding(
            protein_alphabet_len,
            protein_emb_size,
            trainable=True,
            name='protein_emb',
            mask_zero=True,
            embeddings_initializer=initializer)(protein_seq)

        protein_skip_con = tf.keras.layers.Conv1D(protein_emb_size,
                                                  protein_kernel[1],
                                                  strides=protein_strides_down,
                                                  name='protein_skip_con_1',
                                                  kernel_initializer=initializer,
                                                  kernel_regularizer=regularizer,
                                                  activation=activation)(protein_emb)

        nodes = input_nodes
        adj = input_adj

        if graph_type == 'gcn':
            for level in range(graph_depth):
                nodes_h = tf.keras.layers.Dense(num_graph_features*(level+1), name=f'dense_{level+1}', activation='linear')(nodes)
                matmul = tf.keras.layers.Lambda(lambda x: tf.matmul(
                    x[1], x[0], transpose_a=True), output_shape=(None, None, num_graph_features*(level+1)**2))
                nodes_h = matmul((nodes_h, adj))
                nodes = tf.keras.layers.ReLU()(nodes_h)

        elif graph_type == 'gin_eps0':
            # eps = 0
            for level in range(graph_depth):
                nodes_h = tf.keras.layers.Dense(num_graph_features, name=f'dense_{level+1}', activation='linear')(nodes)
                matmul = tf.keras.layers.Lambda(lambda x: tf.matmul(
                    x[1], x[0], transpose_a=True), output_shape=(None, None))
                nodes_h = matmul((nodes_h, adj))
                nodes_h = tf.keras.layers.ReLU()(nodes_h)
                nodes = tf.keras.layers.Add()([nodes, nodes_h])
                nodes = tf.keras.layers.BatchNormalization()(nodes)

        drug_global_pooled = tf.keras.layers.GlobalMaxPooling1D()(nodes)

        for b in range(2, num_res_blocks + 2):
            protein_cnv = self._connv_batchnorm(protein_skip_con,
                                                num_filters_protein_1,
                                                protein_kernel[0], initializer, regularizer,
                                                activation)
            protein_cnv = self._connv_batchnorm(protein_cnv, num_filters_protein_2,
                                                protein_kernel[1], initializer, regularizer, None)
            protein_skip_con = tf.keras.layers.Add(name=f'protein_skip_con_{b}')([protein_skip_con, protein_cnv])

            if activation == 'relu':
                protein_skip_con = tf.keras.layers.ReLU()(protein_skip_con)
            elif activation == 'selu':
                protein_skip_con = tf.keras.activations.selu()(protein_skip_con)

        if pooling_mode == 'max':
            protein_global_pooled = tf.keras.layers.GlobalMaxPool1D(
                name='protein_global_pooled')(protein_skip_con)
        elif pooling_mode == 'avg':
            protein_global_pooled = tf.keras.layers.GlobalAveragePooling1D(
                name='protein_global_pooled')(protein_skip_con)

        drug_protein_vec = tf.keras.layers.concatenate(
            [drug_global_pooled, protein_global_pooled], axis=-1, name='drug_protein_vec')
        drug_protein_vec = tf.keras.layers.BatchNormalization()(drug_protein_vec)
        drug_protein_vec = tf.keras.layers.Dropout(
            dropout_rate)(drug_protein_vec)

        shared_layer = tf.keras.layers.Dense(
            units_per_layer, name='shared_layer_1', activation=activation)(drug_protein_vec)
        shared_layer = tf.keras.layers.BatchNormalization()(shared_layer)
        shared_layer = tf.keras.layers.Dropout(dropout_rate)(shared_layer)

        shared_layer = tf.keras.layers.Dense(
            units_per_layer, name='shared_layer_2', activation=activation)(shared_layer)
        shared_layer = tf.keras.layers.BatchNormalization()(shared_layer)
        shared_layer = tf.keras.layers.Dropout(dropout_rate)(shared_layer)

        head_layers = {}
        for out in order:
            head = tf.keras.layers.Dense(units_per_head,
                                         activation=activation,
                                         name=f'head_{out}_1',
                                         kernel_initializer=initializer)(shared_layer)

            head = tf.keras.layers.Dense(units_per_head,
                                         activation=activation,
                                         name=f'head_{out}_2',
                                         kernel_initializer=initializer)(head)

            head_layers[out] = head

        outputs = []
        for out in order:
            head = head_layers[out]
            output = tf.keras.layers.Dense(1,
                                           activation=activations[out],
                                           name=out,
                                           kernel_initializer=initializer)(head)
            outputs.append(output)

        model = tf.keras.models.Model(inputs=[input_nodes, input_adj, protein_seq, drug_pos],
                                      outputs=outputs)

        model.compile(loss=uselosses,
                      optimizer=optimizer,
                      metrics=usemetrics,
                      loss_weights=loss_weights,
                      run_eagerly=run_eagerly)

        trainable_count = count_params(model.trainable_weights)
        print(f"Done. Total trainable params: {trainable_count}")
        return model
