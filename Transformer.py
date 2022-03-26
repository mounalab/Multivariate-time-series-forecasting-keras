from datetime import datetime
from time import time
import json
import logging

import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import Callback

from kerastuner.tuners import RandomSearch

from sklearn.metrics import r2_score


from utils import rmse, coeff_determination, smape


class Transformer(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self,
        logger=None,
        **kwargs):
        """ Initialization of the RNN Model as TensorFlow computational graph
        """

        with open("parameters.json") as f:
            parameters = json.load(f)


        # Get model hyperparameters
        self.look_back = parameters["look_back"]
        self.n_features = parameters["n_features"]
        self.horizon = parameters["horizon"]

        # Get directories name
        self.log_dir = parameters["log_dir"]
        self.checkpoint_dir = parameters["checkpoint_dir"]

        self.head_size=256
        self.num_heads=4
        self.ff_dim=4
        self.num_transformer_blocks=4
        self.mlp_units=[128]
        self.mlp_dropout=0.4
        self.dropout=0.25


    def transformer_encoder(self,
        inputs):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
        key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)

        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


    def build(self):
        """ Build the model architecture
        """

        inputs = keras.Input(shape=(self.look_back, self.n_features))
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        # output layer
        outputs = layers.Dense(self.horizon)(x)

        return keras.Model(inputs, outputs)

    def restore(self,
        filepath):
        """ Restore a previously trained model
        """

        # Load the architecture
        self.best_model = load_model(filepath, custom_objects={'smape': smape,
                                                         #'mape': mape,
                                                         'rmse' : rmse,
                                                         'coeff_determination' : coeff_determination})

        ## added cause with TF 2.4, custom metrics are not recognize custom metrics with only load-model
        self.best_model.compile(
            optimizer='adam',
            loss = ['mse'],
            metrics=[rmse, 'mae', smape, coeff_determination])


    def train(self,
        X_train,
        y_train,
        epochs=200,
        batch_size=64):
        """ Training the network
        :param X_train: features matrix
        :type 2-D Numpy array of float values
        :param Y_train: one-hot encoded labels matrix
        :type 2-D Numpy array of int values
        :param checkpoint_every: RNN model checkpoint frequency
        :type int
        :param display_step: number of training epochs executed before logging messages
        :type int
        :param verbose: display log messages on screen
        :type boolean
        :return Cost history of each training epoch
        :rtype 1-D Numpy array of floats
        :raises: -
        """

        self.model = self.build()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss = ['mse'],
                           metrics=[rmse, 'mae', smape, coeff_determination],
                           )
        print(self.model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        filepath = self.checkpoint_dir+"/Transformer.best"+datetime.now().strftime('%d%m%Y_%H:%M:%S')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             #validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor, checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])



    def evaluate(self,
        X_test,
        y_test):
        """ Evaluating the network
        :param X_test: features matrix
        :type 2-D Numpy array of float values
        :param Y_test: one-hot encoded labels matrix
        :type 2-D Numpy array of int values
        :return  Evaluation losses
        :rtype 5 Float tuple
        :raise -
        """

        y_pred = self.model.predict(X_test)

        # Print accuracy if ground truth is provided
        """
        if y_test is not None:
            loss_ = session.run(
                self.loss,
                feed_dict=feed_dict)
        """

        _, rmse_result, mae_result, smape_result, _ = self.model.evaluate(X_test, y_test)

        r2_result = r2_score(y_test.flatten(),y_pred.flatten())

        return _, rmse_result, mae_result, smape_result, r2_result
