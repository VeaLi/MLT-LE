# MLT-LE
## Multi-Task Drug Target Affinity Prediction Model



Train, Dev, Test = 733937, 90610, 81549


| Num res. blocks| 1   | 5   | 10  | 15  |
| -------------- | --- | --- | --- | --- |
| Discount       |     |     |     |     |
| -              |     |     |     |     |
| 0.6            |     |     |     |     |


## Install environment

Install dependencies in environment

`conda env create --file conda.yaml`

Activate environment

`source activate mltle`

or

Windows:
Install dependencies in environment

`conda env create --file conda.yaml`

Activate environment

`conda activate mltle`


## Usage

### Training

#### Define model
```python
import mltle as mlt
from collections import defaultdict
import numpy as np
from keras import backend as K

import tensorflow as tf

model = mlt.training.Model(drug_emb_size=64,
              protein_emb_size=32,
              max_drug_len=200,
              drug_alphabet_len=53,
              protein_alphabet_len=8006)

order = ['p1Ki', 'p1IC50', 'p1Kd', 'p1EC50', 'is_active', 'pH', 'pSL']
loss_weights = [1.0] * len(order)

variables = {}
for var in order:
    variables[var] = K.variable(0.0)

LossCallback = mlt.training.LossWithMemoryCallback(variables, discount=0.6)

uselosses = defaultdict(lambda: mlt.training.mse_loss_wrapper)
uselosses['is_active'] = 'binary_crossentropy'

for k, v in variables.items():
    if k not in uselosses.keys():
        uselosses[k] = uselosses[k](v)

usemetrics = {'is_active': tf.keras.metrics.AUC()}

activations = defaultdict(lambda: 'linear')
activations['is_active'] = 'sigmoid'

initializer = tf.keras.initializers.HeUniform(seed=7)
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)

model = model.create_model(order=order,
                    activations=activations,
                    num_res_blocks=3,
                    units_per_head=32,
                    units_per_layer=128,
                    dropout_rate=0.3,
                    drug_kernel=(2, 3),
                    protein_kernel=(7, 7),
                    loss_weights=loss_weights,
                    usemetrics=usemetrics,
                    uselosses=uselosses,
                    initializer=initializer,
                    optimizer=optimizer)

```

#### Prepare data

##### Map strings to integers

- drug_mode:
    - "smiles_1" - map a drug SMILES string to a vector of integers, 
     ngram=1, match every character, example: CCC -> [4,4,4],
    see `mltle.data.maps.smiles_1` for the map

    -  "smiles_2" - map a drug SMILES string to a vector of integers, 
    ngram=2, match every character, example: CCC -> [2,2],
    see `mltle.data.maps.smiles_2` for the map

    -  "selfies_1" - map a drug SELFIES string to a vector of integers, 
    ngram=1, match every character, example: CCC -> [3,3,3],
    see `mltle.data.maps.selfies_1` for the map

    -  "selfies_3" - map a drug SELFIES string to a vector of integers, 
    ngram=3, match every character, example: [C][C] -> [2,2],
    see `mltle.data.maps.selfies_3` for the map

- protein_mode:
    -  "protein_1" - map a protein string to a vector of integers, 
    ngram=1, match every 1 characters, example: LLLSSS -> [3, 3, 3, 5, 5, 5],
    see `mltle.data.maps.protein_1` for the map

    -  "protein_3" - map a protein string to a vector of integers, 
    ngram=3, match every 3 characters, example: LLLSSS -> [1, 3, 13, 2],
    see `mltle.data.maps.protein_3` for the map

```python
mapseq = mlt.datamap.MapSeq(drug_mode='smiles_1',
               				protein_mode='protein_3',
                			max_drug_len=200)

drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()

map_drug, map_protein = mapseq.create_maps(drug_seqs, protein_seqs)
```

#### Create generator

```python
batch_size = 64

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