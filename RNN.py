from datetime import datetime
from time import time
import json
import logging

import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback

from kerastuner.tuners import RandomSearch

from sklearn.metrics import r2_score

from livelossplot import PlotLossesKeras


from utils import rmse, coeff_determination, smape


class RNN(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """

    def __init__(self):
        """ Initialization of the object
        """

        with open("parameters.json") as f:
            parameters = json.load(f)


        # Get model hyperparameters
        self.look_back = parameters["look_back"]
        self.n_features = parameters["n_features"]
        self.n_layers_init = parameters["n_layers_init"]
        self.n_layers_end = parameters["n_layers_end"]
        self.n_units_init = parameters["n_units_init"]
        self.n_units_end = parameters["n_units_end"]
        self.unit = parameters["unit"]
        self.horizon = parameters["horizon"]

        # Get directories name
        self.log_dir = parameters["log_dir"]
        self.checkpoint_dir = parameters["checkpoint_dir"]


    def build(self, hp):
        """ Build the model architecture
        :param hp: hyperparameters tuner
        :type Keras Tuner
        """

        model = Sequential()

        if self.unit=="lstm":
            model.add(layers.LSTM(hp.Int('input_unit', min_value=self.n_units_init, max_value=self.n_units_end, step=16), input_shape=(self.look_back, self.n_features), return_sequences=True))

        else:
            model.add(layers.GRU(hp.Int('input_unit', min_value=self.n_units_init, max_value=self.n_units_end, step=16), input_shape=(self.look_back, self.n_features), return_sequences=True))

        # Tune the number of layers
        for i in range(hp.Int('num_layers', self.n_layers_init, self.n_layers_end)):

            if self.unit=="lstm":
                model.add(layers.LSTM(units=hp.Int(f"units_{i}", self.n_units_init, self.n_units_end, step=16),
                        activation=hp.Choice('act_' + str(i), ['relu', 'tanh']),
                        return_sequences=True))

            else:
                model.add(layers.GRU(units=hp.Int(f"units_{i}", self.n_units_init, self.n_units_end, step=16),
                        activation=hp.Choice('act_' + str(i), ['relu', 'tanh']),
                        return_sequences=True))

        if self.unit=="lstm":
            model.add(layers.LSTM(hp.Int('layer_2_neurons', min_value=self.n_units_init, max_value=self.n_units_end, step=16)))

        else:
            model.add(layers.GRU(hp.Int('layer_2_neurons', min_value=self.n_units_init, max_value=self.n_units_end, step=16)))

        model.add(layers.Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))

        model.add(layers.Dense(self.horizon))
        model.compile(optimizer='adam', loss = ['mse'], metrics=[rmse, 'mae', smape, coeff_determination])


        return model

    def restore(self,
        filepath):
        """ Restore a previously trained model
        :param filepath: path to saved model
        :type str
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
        batch_size=32):
        """ Training the network
        :param X_train: training feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_train: training target vectors
        :type 2-D Numpy array of float values
        :param epochs: number of training epochs
        :type int
        :param batch_size: size of batches used at each forward/backward propagation
        :type int
        :return -
        :raises: -
        """

        # Use Keras tuner for automated hyperparameters tuning
        tuner = RandomSearch(
            self.build,
            objective = 'val_loss',
            max_trials = 5,
            executions_per_trial = 3,
            directory='ktuner',
            project_name='kerastuner_bayesian_cnn',
            overwrite=True,
            )

        """
        n_samples = X_train.shape[0]
        split = int(n_samples*0.8)

        X_train_val = X_train[:split,:,:]
        y_train_val = y_train[:split,:]
        X_val = X_train[split:,:,:]
        y_val = y_train[split:,:]"""

        #tuner.search(X_train_val, y_train_val, epochs=5,  validation_data=(X_val,y_val))
        tuner.search(X_train, y_train, epochs=5, validation_split=0.2)
        print(tuner.search_space_summary())

        self.best_model = tuner.get_best_models()[0]
        print(self.best_model.summary())

        # Stop training if error does not improve within 50 iterations
        early_stopping_monitor = EarlyStopping(patience=50, restore_best_weights=True)

        # Save the best model ... with minimal error
        filepath = self.checkpoint_dir+"/RNN.best"+datetime.now().strftime('%d%m%Y_%H:%M:%S')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callback_history = self.best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                             #validation_split=0.2,
                             verbose=1,
                             callbacks=[early_stopping_monitor, checkpoint])
                             #callbacks=[PlotLossesKeras(), early_stopping_monitor, checkpoint])



    def evaluate(self,
        X_test,
        y_test):
        """ Evaluating the network
        :param X_test: test feature vectors [#batch,#number_of_timesteps,#number_of_features]
        :type 3-D Numpy array of float values
        :param Y_test: test target vectors
        :type 2-D Numpy array of int values
        :return  Evaluation losses
        :rtype 5 Float tuple
        :raise -
        """

        y_pred = self.best_model.predict(X_test)

        # Print accuracy if ground truth is provided
        """
        if y_test is not None:
            loss_ = session.run(
                self.loss,
                feed_dict=feed_dict)
        """

        _, rmse_result, mae_result, smape_result, _ = self.best_model.evaluate(X_test, y_test)

        r2_result = r2_score(y_test.flatten(),y_pred.flatten())

        return _, rmse_result, mae_result, smape_result, r2_result
