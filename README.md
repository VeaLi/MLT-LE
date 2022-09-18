## MLT-LE: predicting drug–target binding affinity with multi-task residual neural networks

This repository hosts the MLT-LE framework, designed to provide a set of tools for creating significantly deep multi-task models for predicting drug-target affinity.

### Features
The MLT-LE framework has a variety of encoding options, including: __variable-length encoding__ or __positional encoding__ with different __label encodings__, as well as __graph-based encodings__ with different graph normalization options (See description below in [Prepare data](#prepare-data) section). All models are implemented as residual networks, which allows for significantly deep models of 15 layers or more.

In addition, the framework allows a built-in dimensionality reduction/expansion using __adjustable convolution step__ for protein sequences and __deconvolution__ for drug sequences.

### Performance

:Performance comparisons on BindingDB EC50 test set:
|Method | Protein rep. |   Compound rep. |  CI | MSE |
|-------|--------------|-----------------|-----|-----|
| GraphDTA (Nguyen et al., 2021)  |  CNN  |  GCN |    0.805   | 0.690 | 
| GraphDTA (Nguyen et al., 2021)  |    CNN  |  GAT_GCN |   0.808 |   0.676|  
| GraphDTA (Nguyen et al., 2021)  |    CNN  | GAT| 0.801    |  0.829  | 
| GraphDTA (Nguyen et al., 2021)   |   CNN |   GIN  |   0.813  |  0.648 |
| MLT-LE  |   CNN |   CNN |   0.817  |  0.657 | 
| MLT-LE   |   CNN |   GCN |   0.823  | 0.599  | 
| MLT-LE   |   CNN |   GIN |   0.820  |  0.615 | 


Performance comparisons on Davis test set
|Method | Protein rep. |   Compound rep. |  CI | MSE |
|-------|--------------|-----------------|-----|-----|
|DeepDTA (Öztürk et al., 2018) |  Smith-Waterman  |Pubchem-Sim    | 0.790  | 0.608 |
|DeepDTA (Öztürk et al., 2018)  | Smith-Waterman  | CNN |  0.886 |  0.420 |
| DeepDTA (Öztürk et al., 2018) |  CNN | Pubchem-Sim    |  0.835  |  0.419 | 
| KronRLS (Cichonska et al., 2017, 2018)  | Smith-Waterman|   Pubchem-Sim    |  0.871 |   0.379 | 
| SimBoost (He et al., 2017)|   Smith-Waterman |  Pubchem-Sim |     0.872 |   0.282 | 
| DeepDTA (Öztürk et al., 2018)  |  CNN  | CNN |  0.878|    0.261|  
| WideDTA (Öztürk et al., 2019)   | CNN + PDM  |   1D + LMCS  |  0.886  |  0.262 | 
| GraphDTA (Nguyen et al., 2021)  |  CNN  |  GCN |    0.880   | 0.254 | 
| GraphDTA (Nguyen et al., 2021)  |    CNN  |  GAT_GCN |   0.881 |   0.245|  
| GraphDTA (Nguyen et al., 2021)  |    CNN  | GAT|    0.892  |  0.232 | 
| GraphDTA (Nguyen et al., 2021)   |   CNN |   GIN  |   0.893  |  0.229 | 
| MLT-LE   |   CNN |   GIN |   0.884  |  0.235 | 
]

** Davis performance table for GraphDTA adaptated from (Nguyen et al., 2021).

## Install environment

Install dependencies and activate environment

```sh
conda env create --file mltle.yml
source activate mltle
```

or:

```sh
conda env create --file mltle.yml
conda activate mltle
```

### Key dependencies
  - tensorflow==2.9.1
  - rdkit-pypi==2022.3.5
  - networkx==2.8.5

See all dependencies in `mltle.yml` file.


## Usage

- [Train](#train)
    - [Define model](#define-model)
    - [Prepare data](#prepare-data)
- [Predict](#predict)

### Train

#### Define model
```python
NUM_RES_BLOCKS = 1
GRAPH_DEPTH = 4
GRAPH_FEATURES = 'g78'
GRAPH_TYPE = 'gcn'

model = mlt.graph_training.GraphModel(protein_emb_size=64, protein_alphabet_len=8006)

order = ['pKi', 'pIC50', 'pKd', 'pEC50', 'is_active', 'qed', 'pH']
loss_weights = [1.0] * len(order)

variables = {}
for var in order:
    variables[var] = K.variable(0.0)

LossCallback = mlt.training_utils.LossWithMemoryCallback(
    variables, discount=DISCOUNT, decay=0.8)

uselosses = defaultdict(lambda: mlt.training_utils.mse_loss_wrapper)
uselosses['is_active'] = 'binary_crossentropy'
uselosses['qed'] = 'binary_crossentropy'

for k, v in variables.items():
    if k not in uselosses.keys():
        uselosses[k] = uselosses[k](v)

usemetrics = {data_type: [tf.keras.metrics.mse, mlt.training_utils.cindex_score]}

activations = defaultdict(lambda: 'linear')
activations['is_active'] = 'sigmoid'
activations['qed'] = 'sigmoid'

initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='normal', seed=SEED)
optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.Nadam(), sync_period=3)

model = model.create_model(order=order,
                           activations=activations,
                           activation='relu',
                           pooling_mode='max',
                           num_res_blocks=NUM_RES_BLOCKS,
                           units_per_head=64,
                           units_per_layer=1024,
                           dropout_rate=0.3,
                           protein_kernel=(7, 7),
                           loss_weights=loss_weights,
                           usemetrics=usemetrics,
                           uselosses=uselosses,
                           initializer=initializer,
                           optimizer=optimizer,
                           protein_strides_down=1,
                           graph_depth=GRAPH_DEPTH,
                           num_graph_features=78,
                           graph_type=GRAPH_TYPE)
```

#### Prepare data

For drug sequences, when using CNN-based models for all encodings, you can enable positional encoding or variable length encoding using the `positional = True` or `max_drug_len='inf'` options. Only variable length coding is available for protein sequences: `max_drug_len='inf'`.

##### Available encodings  (SMILES)

|`drug_mode`||
|-----------|-|
|`smiles_1`|map a drug SMILES string to a vector of integers, ngram=1, match every 1 character, example: CCC -> [4,4,4], see `mltle.data.maps.smiles_1` for the map|
|`smiles_2`|map a drug SMILES string to a vector of integers, ngram=2, match every 2 characters, example: CCC -> [2,2], see `mltle.data.maps.smiles_2` for the map|
|`selfies_1`|map a drug SELFIES string to a vector of integers, ngram=1, match every 1 character, example: CCC -> [3,3,3], see `mltle.data.maps.selfies_1` for the map|
|`selfies_3`|map a drug SELFIES string to a vector of integers, ngram=3, match every 3 characters, example: [C][C] -> [2,2], see `mltle.data.maps.selfies_3` for the map|

|`protein_mode`||
|-----------|-|
|`protein_1`|map a protein string to a vector of integers, ngram=1, match every 1 character, example: LLLSSS -> [3, 3, 3, 5, 5, 5], see `mltle.data.maps.protein_1` for the map|
|`protein_3`|map a protein string to a vector of integers,  ngram=3, match every 3 characters, example: LLLSSS -> [1, 3, 13, 2], see `mltle.data.maps.protein_3` for the map|

##### Available encodings (GRPAHS)

###### Graph features set
|`drug_mode`||
|-----------------|-|
|`g24`| Possible scope of atomic features in BindingDB, results in embedding of length=24|
|`g78`| GraphDTA possible features set (Nguyen *et al*, 2021), results in embedding of length=78, produces better results than `g24`|

###### Graph type
|`graph_type`||
|------------|-|
|`gcn`| Graph Convolution Network based drug embedding|
|`gin_eps0`| Graph Isomorphism Network based drug-embedding with `eps=0`|


##### Available graph normalization (GRAPHS)
|`graph_normalization_type`||
|------------|-|
|`''`| `None` - skip normalization, for example for `GIN`-based encoding|
|`kipf`| use (Kipf and Welling, 2017) normalization|
|`laplacian`| use Laplacian normalization|

##### Example of creating encoding (SMILES)

```python
mapseq = mlt.datamap.MapSeq(drug_mode='smiles_1',
                            protein_mode='protein_3',
                            max_drug_len=200,
                            max_protein_len=1000) # or 'inf' for variable length

drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()

map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)
```

##### Example of creating encoding (GRAPHS)
###### GCN
```python
mapseq = mlt.datamap.MapSeq(drug_mode='g78',
                            protein_mode='protein_3',
                            max_drug_len=100,
                            max_protein_len=1000,
                            graph_normalize=True,
                            graph_normalization_type='kipf')

drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()
map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)
```
###### GIN
```python
mapseq = mlt.datamap.MapSeq(drug_mode='g78',
                            protein_mode='protein_3',
                            max_drug_len=100,
                            max_protein_len=1000,
                            graph_normalize=False,
                            graph_normalization_type='')

drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()
map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)
```

##### Example of creating data generator (SMILES)

```python
batch_size = 128

train_gen = mlt.datagen.DataGen(X_train, map_drug, map_protein)
train_gen = train_gen.get_generator(batch_size)

valid_gen = mlt.datagen.DataGen(X_valid, map_drug, map_protein)
valid_gen = valid_gen.get_generator(batch_size)

test_gen = mlt.datagen.DataGen(X_test,
                               map_drug,
                               map_protein,
                               shuffle=False,
                               test_only=True)

test_gen = test_gen.get_generator(test_batch_size)
```

##### Example of creating data generator (GRAPHS)

```python
batch_size = 128

batch_size = BATCH_SIZE

train_gen = mlt.datagen.DataGen(X_train, map_drug, map_protein, drug_graph_mode=True)
train_gen = train_gen.get_generator(batch_size)

valid_gen = mlt.datagen.DataGen(X_valid, map_drug, map_protein, drug_graph_mode=True, shuffle=False)
valid_gen = valid_gen.get_generator(batch_size)

test_gen = mlt.datagen.DataGen(X_test,
                               map_drug,
                               map_protein,
                               shuffle=False,
                               test_only=True, 
                               drug_graph_mode=True)

test_gen = test_gen.get_generator(test_batch_size)
```


### Predict
Examples of training and prediction for each available model can be found in the folder `examples/graphdta-mltle-test/mltle/notebooks`.

### For citation
___
If you found this package useful, you can find our work here: [http://arxiv.org/abs/2209.06274](https://arxiv.org/abs/2209.06274).
```
@misc{vinogradova2022mltle,
    title = {{MLT}-{LE}: predicting drug-target binding affinity with multi-task residual neural networks},
    url = {http://arxiv.org/abs/2209.06274},
    doi = {10.48550/arXiv.2209.06274},
    urldate = {2022-09-15},
    publisher = {arXiv},
    author = {Vinogradova, Elizaveta and Pats, Karina and Molnár, Ferdinand and Fazli, Siamac},
    month = sep,
    year = {2022}
}
```