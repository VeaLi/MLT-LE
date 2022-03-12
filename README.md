# MLT-LE
 MLT-LE or Multi-Task Drug Target Affinity Prediction Model

MLT-LE is method for predicting the affinity of ligands to targets, which allows to cope with the noise in the raw data. It uses the multi-task learning paradigm.

Currently, there are a small number of models that use multi-task learning to solve the problem of predicting ligand-target binding strength. Two recently developed models, Multi-PLI (2021) and GanDTI (2021), emphasize the use of a multi-target approach to simultaneously solve classification and regression problems (joint-task learning), with different binding strength measures (Kd, Ki, IC50, EC50) separated in the learning process. In contrast, in this project, all measures of affinity (binding strength) are used simultaneously in training. This allows an order of magnitude more data to be used, which is especially important because some ligand-protein pairs are unevenly represented for different affinity measures.

The developed model has 2 inputs and 5 outputs, depending on the number of predicted values. The model takes as input information about the structure of ligands and proteins in the form of text strings. The output is a prediction of five values for such pairs (ligand-protein) - different measures of affinity (Kd, Ki, IC50, EC50) and probability of belonging to a set of active structures.
In this work, data from the BindingDB database containing records of experimental results on the binding strength of ligands to target proteins were used to train and test the model's ability to predict binding strength.

The model proposed in this paper not only predicts various indicators of biological activity simultaneously, but is also capable of learning from the full amount of available data and filling in missing values during the learning and prediction process. According to the results, the use of a multi-task approach makes the developed model more suitable for evaluating *de novo* generative models and generated molecules. The model is also comparable in performance to benchmark models (DeepDTA, GraphDTA).


![performance](many-rankings.png)