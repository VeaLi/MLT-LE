## Multi-Task Drug Target Affinity Prediction Model


### Data
---

| Split | Samples |
| ----- | ------- |
| Train | 580,236 |
| Dev   | 71,635  |
| Test  | 64,471  |
| *__Total__* | 716,342 |
Table 1: Num. of non-overlaping samples (unique drug-target pairs) in train-dev-test

Balance for each class: <br/>
`1.0` : 67.4% records (active compounds)<br/>
`0.0` : 33.6% records (non-active compounds)<br/>



## Install environment

Install dependencies and activate environment

```sh
conda env create --file conda.yaml
source activate mltle
```

or:

```sh
conda env create --file conda.yaml
conda activate mltle
```

## MLFlow project

In MLFlow/ directory MLFlow project can be found.
It is used to analyze the impact of model parameters. For example, the number of residual blocks.

To run MLFlow pipeline:
```bash
conda env create --file conda.yaml
conda activate mltle_params
python run_experiments.py
mlflow ui
```
`mlflow ui` will serve at http://127.0.0.1:5000.

Parameters can be set in `run_experiments.py`. 
Parameters search is performed on subset of data. See `MLFlow/data_subset/`



## Usage

- [Predict](#predict)
- [Train](#train)
    - [Define model](#define-model)
    - [Prepare data](#prepare-data)
      - [Map strings to integers](#map-string-to-integers)
      - [Data generator](#data-generator)


### Predict

```Python
import sys
sys.path.insert(0, '../')

from examples.example_target_sequences import VDR, MTOR, GABA
import pandas as pd
import mltle as mlt

model = mlt.predict.Model('Res2_09_qed_ph')

# this model expect SMILES string to be canonical
vdr_ligand_calcitriol = "C=C1C(=CC=C2CCCC3(C)C2CCC3C(C)CCCC(C)(C)O)CC(O)CC1O"
vdr_ligand_calcitriol = mlt.utils.to_non_isomeric_canonical(vdr_ligand_calcitriol)

gaba_ligand_diazepam = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"
gaba_ligand_diazepam = mlt.utils.to_non_isomeric_canonical(gaba_ligand_diazepam)

mtor_ligand_torin1 = "CCC(=O)N1CCN(c2ccc(-n3c(=O)ccc4cnc5ccc(-c6cnc7ccccc7c6)cc5c43)cc2C(F)(F)F)CC1"
mtor_ligand_torin1 = mlt.utils.to_non_isomeric_canonical(mtor_ligand_torin1)

X_predict = pd.DataFrame()
X_predict['drug'] = [vdr_ligand_calcitriol, gaba_ligand_diazepam, mtor_ligand_torin1]
X_predict['protein'] = [VDR, GABA, MTOR]

model.predict(X_predict)
```
Output:
|    | drug                                                                          | protein                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |    p1Ki |   p1IC50 |    p1Kd |   p1EC50 |   is_active |      qed |      pH |
|---:|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|---------:|--------:|---------:|------------:|---------:|--------:|
|  0 | C=C1C(=CC=C2CCCC3(C)C2CCC3C(C)CCCC(C)(C)O)CC(O)CC1O                           | MEAMAASTSLPDPGDFDRNVPRICGVCGDRATGFHFNAMTCEGCKGFFRRSMKRKALFTCPFNGDCRITKDNRRHCQACRLKRCVDIGMMKEFILTDEEVQRKREMILKRKEEEALKDSLRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSHPSRPNSRHTPSFSGDSSSSCSDHCITSSDMMDSSSFSNLDLSEEDSDDPSVTLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSSAIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIKFQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQTYIRCRHPPPGSHLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKLTPLVLEVFGNEIS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 1.94258 |  3.30496 | 2.14385 |  2.6251  |    0.994911 | 0.344082 | 7.27025 |
|  1 | CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21                                           | MRKSPGLSDCLWAWILLLSTLTGRSYGQPSLQDELKDNTTVFTRILDRLLDGYDNRLRPGLGERVTEVKTDIFVTSFGPVSDHDMEYTIDVFFRQSWKDERLKFKGPMTVLRLNNLMASKIWTPDTFFHNGKKSVAHNMTMPNKLLRITEDGTLLYTMRLTVRAECPMHLEDFPMDAHACPLKFGSYAYTRAEVVYEWTREPARSVVVAEDGSRLNQYDLLGQTVDSGIVQSSTGEYVVMTTHFHLKRKIGYFVIQTYLPCIMTVILSQVSFWLNRESVPARTVFGVTTVLTMTTLSISARNSLPKVAYATAMDWFIAVCYAFVFSALIEFATVNYFTKRGYAWDGKSVVPEKPKKVKDPLIKKNNTYAPTATSYTPNLARGDPGLATIAKSATIEPKEVKPETKPPEPKKTFNSVSKIDRLSRIAFPLLFGIFNLVYWATYLNREPQLKAPTPHQ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 3.8339  |  5.13389 | 4.26874 |  5.68956 |    0.910887 | 0.675182 | 7.35552 |
|  2 | CCC(=O)N1CCN(c2ccc(-n3c(=O)ccc4cnc5ccc(-c6cnc7ccccc7c6)cc5c43)cc2C(F)(F)F)CC1 | MLGTGPAAATTAATTSSNVSVLQQFASGLKSRNEETRAKAAKELQHYVTMELREMSQEESTRFYDQLNHHIFELVSSSDANERKGGILAIASLIGVEGGNATRIGRFANYLRNLLPSNDPVVMEMASKAIGRLAMAGDTFTAEYVEFEVKRALEWLGADRNEGRRHAAVLVLRELAISVPTFFFQQVQPFFDNIFVAVWDPKQAIREGAVAALRACLILTTQREPKEMQKPQWYRHTFEEAEKGFDETLAKEKGMNRDDRIHGALLILNELVRISSMEGERLREEMEEITQQQLVHDKYCKDLMGFGTKPRHITPFTSFQAVQPQQSNALVGLLGYSSHQGLMGFGTSPSPAKSTLVESRCCRDLMEEKFDQVCQWVLKCRNSKNSLIQMTILNLLPRLAAFRPSAFTDTQYLQDTMNHVLSCVKKEKERTAAFQALGLLSVAVRSEFKVYLPRVLDIIRAALPPKDFAHKRQKAMQVDATVFTCISMLARAMGPGIQQDIKELLEPMLAVGLSPALTAVLYDLSRQIPQLKKDIQDGLLKMLSLVLMHKPLRHPGMPKGLAHQLASPGLTTLPEASDVGSITLALRTLGSFEFEGHSLTQFVRHCADHFLNSEHKEIRMEAARTCSRLLTPSIHLISGHAHVVSQTAVQVVADVLSKLLVVGITDPDPDIRYCVLASLDERFDAHLAQAENLQALFVALNDQVFEIRELAICTVGRLSSMNPAFVMPFLRKMLIQILTELEHSGIGRIKEQSARMLGHLVSNAPRLIRPYMEPILKALILKLKDPDPDPNPGVINNVLATIGELAQVSGLEMRKWVDELFIIIMDMLQDSSLLAKRQVALWTLGQLVASTGYVVEPYRKYPTLLEVLLNFLKTEQNQGTRREAIRVLGLLGALDPYKHKVNIGMIDQSRDASAVSLSESKSSQDSSDYSTSEMLVNMGNLPLDEFYPAVSMVALMRIFRDQSLSHHHTMVVQAITFIFKSLGLKCVQFLPQVMPTFLNVIRVCDGAIREFLFQQLGMLVSFVKSHIRPYMDEIVTLMREFWVMNTSIQSTIILLIEQIVVALGGEFKLYLPQLIPHMLRVFMHDNSPGRIVSIKLLAAIQLFGANLDDYLHLLLPPIVKLFDAPEAPLPSRKAALETVDRLTESLDFTDYASRIIHPIVRTLDQSPELRSTAMDTLSSLVFQLGKKYQIFIPMVNKVLVRHRINHQRYDVLICRIVKGYTLADEEEDPLIYQHRMLRSGQGDALASGPVETGPMKKLHVSTINLQKAWGAARRVSKDDWLEWLRRLSLELLKDSSSPSLRSCWALAQAYNPMARDLFNAAFVSCWSELNEDQQDELIRSIELALTSQDIAEVTQTLLNLAEFMEHSDKGPLPLRDDNGIVLLGERAAKCRAYAKALHYKELEFQKGPTPAILESLISINNKLQQPEAAAGVLEYAMKHFGELEIQATWYEKLHEWEDALVAYDKKMDTNKDDPELMLGRMRCLEALGEWGQLHQQCCEKWTLVNDETQAKMARMAAAAAWGLGQWDSMEEYTCMIPRDTHDGAFYRAVLALHQDLFSLAQQCIDKARDLLDAELTAMAGESYSRAYGAMVSCHMLSELEEVIQYKLVPERREIIRQIWWERLQGCQRIVEDWQKILMVRSLVVSPHEDMRTWLKYASLCGKSGRLALAHKTLVLLLGVDPSRQLDHPLPTVHPQVTYAYMKNMWKSARKIDAFQHMQHFVQTMQQQAQHAIATEDQQHKQELHKLMARCFLKLGEWQLNLQGINESTIPKVLQYYSAATEHDRSWYKAWHAWAVMNFEAVLHYKHQNQARDEKKKLRHASGANITNATTAATTAATATTTASTEGSNSESEAESTENSPTPSPLQKKVTEDLSKTLLMYTVPAVQGFFRSISLSRGNNLQDTLRVLTLWFDYGHWPDVNEALVEGVKAIQIDTWLQVIPQLIARIDTPRPLVGRLIHQLLTDIGRYHPQALIYPLTVASKSTTTARHNAANKILKNMCEHSNTLVQQAMMVSEELIRVAILWHEMWHEGLEEASRLYFGERNVKGMFEVLEPLHAMMERGPQTLKETSFNQAYGRDLMEAQEWCRKYMKSGNVKDLTQAWDLYYHVFRRISKQLPQLTSLELQYVSPKLLMCRDLELAVPGTYDPNQPIIRIQSIAPSLQVITSKQRPRKLTLMGSNGHEFVFLLKGHEDLRQDERVMQLFGLVNTLLANDPTSLRKNLSIQRYAVIPLSTNSGLIGWVPHCDTLHALIRDYREKKKILLNIEHRIMLRMAPDYDHLTLMQKVEVFEHAVNNTAGDDLAKLLWLKSPSSEVWFDRRTNYTRSLAVMSMVGYILGLGDRHPSNLMLDRLSGKILHIDFGDCFEVAMTREKFPEKIPFRLTRMLTNAMEVTGLDGNYRITCHTVMEVLREHKDSVMAVLEAFVYDPLLNWRLMDTNTKGNKRSRTRTDSYSAGQSVEILDGVELGEPAHKKTGTTVPESIHSFIGDGLVKPEALNKKAIQIINRVRDKLTGRDFSHDDTLDVPTQVELLIKQATSHENLCQCYIGWCPFW | 3.11253 |  4.0077  | 3.71179 |  5.09542 |    0.96847  | 0.330529 | 7.34997 |


### Train

#### Define model
```python
# necessary imports
import tensorflow as tf
from keras import backend as K
import numpy as np

import mltle as mlt

# additionally
from collections import defaultdict
import tensorflow_addons as tfa

tf.keras.backend.clear_session()
np.random.seed(SEED)
tf.random.set_seed(SEED)

model = mlt.training.Model(drug_emb_size=128,
              protein_emb_size=64,
              max_drug_len=200,
              drug_alphabet_len=53,
              protein_alphabet_len=8006)

order = ['p1Ki', 'p1IC50', 'p1Kd', 'p1EC50', 'is_active', 'qed', 'pH']
loss_weights = [1.0] * len(order)

variables = {}
for var in order:
    variables[var] = K.variable(0.0)

LossCallback = mlt.training.LossWithMemoryCallback(variables, discount=DISCOUNT, decay = 0.8)

uselosses = defaultdict(lambda: mlt.training.mse_loss_wrapper)
uselosses['is_active'] = 'binary_crossentropy'
uselosses['qed'] = 'binary_crossentropy'

for k, v in variables.items():
    if k not in uselosses.keys():
        uselosses[k] = uselosses[k](v)

usemetrics = {'is_active': tf.keras.metrics.AUC()}

activations = defaultdict(lambda: 'linear')
activations['is_active'] = 'sigmoid'
activations[ 'qed']  = 'sigmoid'

initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='normal', seed=SEED)
optimizer = tfa.optimizers.Lookahead(tf.keras.optimizers.Nadam(), sync_period=3)

model = model.create_model(order=order,
                            activations=activations,
                            activation = 'relu',
                            pooling_mode = 'max',
                            num_res_blocks=NUM_RES_BLOCKS,
                            units_per_head=64,
                            units_per_layer=512,
                            dropout_rate=0.3,
                            drug_kernel=(2, 3),
                            protein_kernel=(7, 7),
                            loss_weights=loss_weights,
                            usemetrics=usemetrics,
                            uselosses=uselosses,
                            initializer=initializer,
                            optimizer=optimizer,
                            drug_strides_up=1,
                            protein_strides_down=1,
                            positional=False)
```

#### Prepare data

##### Map strings to integers

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


```python
mapseq = mlt.datamap.MapSeq(drug_mode='smiles_1',
                            protein_mode='protein_3',
                            max_drug_len=200,
                            max_protein_len=1000) # or 'inf' for variable length

drug_seqs, protein_seqs = data['smiles'].unique(), data['target'].unique()

map_drug, map_protein = mapseq.create_maps(drug_seqs = drug_seqs, protein_seqs = protein_seqs)
```

##### Data generator

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
