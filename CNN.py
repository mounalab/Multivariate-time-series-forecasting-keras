from datetime import datetime
from time import time
import json
import logging

import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback

from kerastuner.tuners import RandomSearch

from sklearn.metrics import r2_score

from livelossplot import PlotLossesKeras



from metrics_utils import rmse, coeff_determination, smape


class CNN(object):
    """ Building the Convolutional Neural Network for Multivariate time series forecasting
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
        """

        model = Sequential()

        #adding first convolutional layer
        model.add(layers.Conv1D(
            #adding filter
            filters=hp.Int('conv_1_filter', min_value=32, max_value=256, step=16),
            # adding filter size or kernel size
            kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
            #activation function
            activation='relu',
            input_shape=(self.look_back, self.n_features)))

        # adding convolutional layers
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(layers.Conv1D(
                filters=hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice('conv_1_kernel', values = [3,5])))

            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling1D(pool_size=2))


        model.add(layers.Flatten())

        # adding fully_conneted n_layers
        for i in range(hp.Int('n_connections', 1, 4)):
            model.add(layers.Dense(hp.Choice(f'n_nodes',
                                  values=[128, 256, 512, 1024])))
            model.add(layers.Activation('relu'))


        # output layer
        layers.Dense(self.horizon)

        #compilation of model
        model.compile(optimizer='adam', loss = ['mse'], metrics=[rmse, 'mae', smape, coeff_determination])

        return model


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
        batch_size=32):
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

        # Use Keras tuner for automated hyperparameters tuning
        tuner = RandomSearch(
            self.build,
            objective = 'loss',
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
        filepath = self.checkpoint_dir+"/CNN.best"+datetime.now().strftime('%d%m%Y_%H:%M:%S')+".hdf5"
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
        :param X_test: features matrix
        :type 2-D Numpy array of float values
        :param Y_test: one-hot encoded labels matrix
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
