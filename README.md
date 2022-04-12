# MLT-LE
## MLT-LE or Multi-Task Drug Target Affinity Prediction Models

MLT-LE is a method for estimating binding strength for drug-target pair, which allows dealing with noise in raw data. It uses the paradigm of multi-task learning.

Drug discovery involves assessing the binding strength between drugs and targets. Obtaining these data experimentally is time-consuming and expensive. Therefore, computational virtual screening methods for predicting binding strength are being widely developed. However, the experimental data used to train these prediction models are very inconsistent and unevenly represented for different drug-target pairs. This leads to biased algorithms that do not generalize well to new data. Therefore, approaches have begun to be developed to address this problem. One of them is the use of multitask learning. However, the extent of the application of this technique in this field is currently heavily undeveloped. The approach proposed in present work hopes to fill this gap. The work explores new possibilities that have not been considered before: using all available data from a single heterogeneous source (as opposed to several homogeneous ones), dealing with missing data, and adding auxiliary tasks to boost performance. In addition, this work suggests different type of data preparation as well as loss function adjustments and a comparison of different architectures for this task.

## Main requirements

- For *basic* models - TensorFlow only.
- For others + TensorFlow Addons.

If you have any troubles try to update TensorFlow.

## Models

Train size: ~ 733937 records<br/>
Valid size: ~ 90610 records<br/>
Test size: ~ 81549 records<br/>
Tatal: ~906096 unique drug-target pairs <br/>



Balance for each class: <br/>
1.0 : 58.3% records<br/>
0.0 : 41.6% records<br/>

All basic CNN models are the same model with the same initialization, just trained differently on the same data. The total number of trainable parameters for this model is just 973,062.

For 50 epochs:

| Model                 | AUC on test | Kd CI | Ki CI | IC50 CI | EC50 CI | Total loss |
| --------------------- | ----------- | ----- | ----- | ------- | ------- | ---------- |
| CNN basic             | 89%         | 78.4  |  77.1 |  79.9   | 81.4    | 5.31       |
| CNN basic with memory | 91%         | 82.6  |  80.1 |  82.2   | 83.5    | 4.20       |
| CNN weights           |             |       |       |         |         |            |

        




## Usage

See corresponding Jupyter Notebook.

## Performance
### CNN basic


| ![performance1](images/many-rankings.png) | 
|:--:| 
| *Ranking comparison* |

| ![performance2](images/auc_cnn_basic.png) | 
|:--:| 
| *AUC on test set* |


### CNN basic with memory


## About
Currently, there are a small number of models that use multi-task learning to solve the problem of predicting ligand-target binding strength. Two recently developed models, Multi-PLI (2021) and GanDTI (2021), emphasize the use of a multi-target approach to simultaneously solve classification and regression problems (joint-task learning), separating different binding strength measures (Kd, Ki, IC50, EC50) in the learning process, not using auxiliary tasks, and not dealing with missing data. In contrast, in this project, all affinity measures (binding strength) are used simultaneously in training, as well as some auxiliary data, and missing values are masked. This allows an order of magnitude more data to be used, which is especially important since some ligand-protein pairs are unevenly represented for different affinity measures.

For this work, data from the BindingDB database, which contains records of experimental results on the binding strength of ligands to target proteins, were used to train and test the model's ability to predict binding strength. The model takes as input information about the structure of ligands and proteins in the form of text strings, using simple approaches to represent them that do not require complex dependencies. The output is a prediction of five values for such pairs (ligand-protein) - various affinity measures (Kd, Ki, IC50, EC50) and estimation of the probability of belonging to a group of active structures, as well as a __number of various auxiliary tasks__ (pH, drug mass).

The model proposed here not only predicts different indicators of binding strength simultaneously, but is also able to learn from the entire set of available data and fill in missing values during the learning and prediction process. According to the results, the use of such a multi-task approach makes the developed model suitable for the evaluation of *de novo* generative models and generated molecules. The model is also comparable in performance to benchmark models (GraphDTA library).

## In addition
In this work we use another log transformation (p1Kd = log(Kd+1)) which has no upper bound. Now we test whether this log transformation is better than the standard log transformation: pKd = -log10(Kd*1e-9 + 1e-10). In any case, our transformation is reversible and all functions for the transformation from p1Kd to pKd are shown and available.