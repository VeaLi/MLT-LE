import tensorflow as tf
import numpy as np
from keras import backend as K
import scikitplot as skplt
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy import stats


def get_scores(y_true, y_pred):
    mse = np.round(mean_squared_error(y_true, y_pred), 3)
    rmse = np.round(mse**0.5, 3)
    ci = np.round(concordance_index(y_true, y_pred), 3)
    pearson = np.round(stats.pearsonr(y_true, y_pred)[0], 3)
    spearman = np.round(stats.spearmanr(y_true, y_pred)[0], 3)
    res = f"rmse={rmse}, mse={mse},\npearson={pearson}, spearman={spearman},\nci={ci}"
    return res


def get_batch_size(S):
    mbs = 1
    for i in range(1, min(128, S)):
        if S % i == 0:
            mbs = i
    assert S % mbs == 0

    return mbs


def cindex_score(y_true, y_pred):
    """
    https://stackoverflow.com/questions/43576922/keras-custom-metric-iteration
    Coincides with lifelines.utils.concordance_index
    """
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)


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

    def __init__(self, variables, discount=0.6, decay=0.8):
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
