import tensorflow as tf
import numpy as np


class DataGen:
    """
    This is data generator. 
    This generator expects variable length drug and protein inputs.
    Data should be passed as pandas.DataFrame, 
    columns order should be the following:
        [drug_sequence, protein_sequence, target_1, target_2, ... target_n]
        [    Str,              Str,        Float,    Float,   ...  Float]
        ...

    Target variables can contain both Float and np.nan

    Expects any number of target variables >= 1.


    It is necessary to pass the same number of columns for both train and test.
    For the test, all target variables can be np.nan.


    Parameters
    ----------
    data : pandas.DataFrame
        columns order should be the following:
        [drug_sequence, protein_sequence, target_1, target_2, ... target_n]
        [    Str,              Str,        Float,    Float,   ...  Float]
        ...

        Target variables can contain both Float and np.nan

        Expects any number of target variables >= 1.

    map_drug: Dict[Str, List[int]]
        maps drug string to array of integers

    map_protein: Dict[Str, List[int]]
        maps protein sequence string to array of integers

    shuffle: Bool, default=True
        shuffle data or not

    test_only: Bool, default=False
        sets target variables to 0
    """

    def __init__(self,
                 data,
                 map_drug,
                 map_protein,
                 shuffle=True,
                 test_only=False,
                 drug_graph_mode=False):
        self.data = data
        self.map_drug = map_drug
        self.map_protein = map_protein
        self.shuffle = shuffle
        self.test_only = test_only
        self.drug_graph_mode = drug_graph_mode
        self.size = self.data.shape[0]
        self.inds = list(range(self.size))
        if self.shuffle:
            np.random.shuffle(self.inds)

        self.gen = self._get_inputs()

    def _get_inputs(self):
        """
        This is inner generator.
        Generates one sample
        """

        seen = 0
        while seen < self.size:
            ind = self.inds[seen]
            sample = self.data.iloc[ind, :].values.tolist()

            if not self.drug_graph_mode:
                sample[0] = self.map_drug[sample[0]]
                sample[1] = self.map_protein[sample[1]]
            else:
                nodes, adj = self.map_drug[sample[0]]
                sample[0] = tf.sparse.to_dense(adj)
                sample[1] = self.map_protein[sample[1]]
                sample = [tf.sparse.to_dense(nodes)] + sample

            yield sample
            seen += 1
            if seen == self.size:
                if self.shuffle:
                    np.random.shuffle(self.inds)
                seen = 0

    def get_generator(self, batch_size):
        """
        This is outer generator.
        Generates one batch


        Parameters
        ----------
        batch_size: Int


        Returns
        ----------
            Tuple[List[array_like], List[arry_like]]
            (X, Y) tuple

        """

        while True:
            BATCH = []
            for _ in range(batch_size):
                sample = next(self.gen)
                for k, value in enumerate(sample):
                    if len(BATCH) < (k + 1):
                        BATCH.append([])
                    BATCH[k].append(value)

            if not self.drug_graph_mode:
                drug_lens = [len(d) for d in BATCH[0]]
                drug_len = max(drug_lens)

                target_lens = [len(d) for d in BATCH[1]]
                target_len = max(target_lens)

                # here batch is cut/padded to default maximum length in batch
                # -> batches have different shapes
                BATCH[0] = tf.keras.preprocessing.sequence.pad_sequences(
                    BATCH[0], drug_len, padding='post', truncating='post')
                BATCH[1] = tf.keras.preprocessing.sequence.pad_sequences(
                    BATCH[1], target_len, padding='post', truncating='post')

                # positional encoding, 0 - reserved for padding
                drug_inds = [range(1, v + 1) for v in drug_lens]
                drug_inds = tf.keras.preprocessing.sequence.pad_sequences(
                    drug_inds, drug_len, padding='post', truncating='post')

                # all columns from 3d (index=2), are treated as target variables
                for k in range(2, len(BATCH)):
                    BATCH[k] = np.array(BATCH[k]).flatten()

                if not self.test_only:
                    yield [BATCH[0], BATCH[1], drug_inds], [BATCH[k] for k in range(2, len(BATCH))]
                else:
                    # it is necessary to pass same shaped tuple to tensorflow model
                    # here target variables are zeroes
                    yield [BATCH[0], BATCH[1], drug_inds], [BATCH[k] * 0 for k in range(2, len(BATCH))]

            else:
                target_lens = [len(d) for d in BATCH[2]]
                target_len = max(target_lens)

                # here batch is cut/padded to default maximum length in batch
                # -> batches have different shapes

                BATCH[0] = tf.stack(BATCH[0])
                BATCH[1] = tf.stack(BATCH[1])
                BATCH[2] = tf.keras.preprocessing.sequence.pad_sequences(
                    BATCH[2], target_len, padding='post', truncating='post')

                # positional encoding, 0 - reserved for padding
                drug_inds = [range(10) for v in range(batch_size)]
                drug_inds = tf.keras.preprocessing.sequence.pad_sequences(
                    drug_inds, 10, padding='post', truncating='post')

                # all columns from 3d (index=2), are treated as target variables
                for k in range(3, len(BATCH)):
                    BATCH[k] = np.array(BATCH[k]).flatten()

                if not self.test_only:
                    yield [BATCH[0], BATCH[1], BATCH[2], drug_inds], [BATCH[k] for k in range(3, len(BATCH))]
                else:
                    # it is necessary to pass same shaped tuple to tensorflow model
                    # here target variables are zeroes
                    yield [BATCH[0], BATCH[1], BATCH[2], drug_inds], [BATCH[k] * 0 for k in range(3, len(BATCH))]
