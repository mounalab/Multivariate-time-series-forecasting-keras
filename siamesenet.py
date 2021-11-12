
from livelossplot import PlotLossesKeras

from datetime import datetime
from time import time

import keras
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import TimeDistributed, GRU
from keras.layers import Dense, Conv1D, MaxPool2D, MaxPooling1D, Flatten, Dropout, RepeatVector
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.layers.merge import concatenate
from keras.layers import Input

from data_utils import euclidean_distance, eucl_dist_output_shape


class SiameseNetwork:
    """ Siamese Neural Network class
    """

    def __init__(self, encoder_model):

        # Set encoding parameters
        self.encoder_model = encoder_model

        # Get input shape from the encoder model
        self.input_shape = self.encoder_model.input_shape[1:]

        # Initialize siamese model
        self.siamese_model = None
        self.__initialize_siamese_model()


    def __initialize_siamese_model(self):
        """
        Create the siamese model structure using the supplied encoder model.
        """

        # Define the tensors for the two input images
        left_input = Input(shape=self.input_shape, name="left_input")
        right_input = Input(shape=self.input_shape, name="right_input")

        # Generate the encodings (feature vectors) for the two inputs (left and right)
        encoded_l = self.encoder_model(left_input)
        encoded_r = self.encoder_model(right_input)


        # L2 distance layer between the two encoded outputs
        l2_distance_layer = Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)

        l2_distance = l2_distance_layer([encoded_l, encoded_r])

        # Similarity measure prediction
        prediction = Dense(units=1)(l2_distance)

        self.siamese_model = Model(inputs=[left_input, right_input], outputs=prediction)


    def compile(self, *args, **kwargs):
        """
        Configures the model for training.
        Passes all arguments to the underlying Keras model compile function.
        """
        self.siamese_model.compile(*args, **kwargs)


    def fit_data_generator(self, x_train, y_train, x_test, y_test, batch_size, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the siamese network generator function.
        :param x_train: Training input data.
        :param y_train: Training output data.
        :param x_test: Validation input data.
        :param y_test: Validation output data.
        :param batch_size: Number of pairs to generate per batch.
        """
        train_generator = self.__pair_generator(x_train, y_train, batch_size)
        train_steps = max(len(x_train) / batch_size, 1)
        test_generator = self.__pair_generator(x_test, y_test, batch_size)
        test_steps = max(len(x_test) / batch_size, 1)
        self.siamese_model.fit_generator(train_generator,
                                         steps_per_epoch=train_steps,
                                         validation_data=test_generator,
                                         validation_steps=test_steps,
                                         *args, **kwargs)


    def fit(self, *args, **kwargs):
        """
        Trains the model on data generated batch-by-batch using the siamese network generator function.
        Redirects arguments to the fit_generator function.
        """

        x_train_1 = args[0]
        x_train_2 = args[1]
        y_train = args[2]

        validation_split= kwargs.pop('validation_split')
        batch_size = kwargs.pop('batch_size')
        n_epochs = kwargs.pop('n_epochs')
        checkpoint_path = kwargs.pop('checkpoint_path')

        early_stopping_monitor = EarlyStopping(patience=20)

        # This is used to save the best model, currently monitoring val_mape
        filepath = checkpoint_path+"/Siamese.best"+datetime.now().strftime('%d%m%Y_%H:%M')+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        #schedule = step_decay_schedule(initial_lr=1e-5, decay_factor=0.9, step_size=5)


        #.... Siamese
        history_callback = self.siamese_model.fit([x_train_1, x_train_2], y_train, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split,
                                     verbose=1,
                                     callbacks=[#PlotLossesKeras(),
                                     early_stopping_monitor, checkpoint])
                                               #LRTensorBoard(log_dir="log/tb_log")])

    def restore(self, checkpoint_path):
        """
        Restore a previously trained siamese model

        :param checkpoint_path: Path to the checkpoint file.
        """

        n_input = metadata_test_norm.shape[1]

        #data = np.array(metadata_encoded)

        all_data = metadata_norm
        test_data = metadata_test_norm
        train_data = metadata_train_norm

        print(all_data.shape)
        print(test_data.shape)
        print(train_data.shape)

        # Load saved model
        trained_siamese_model = load_model(checkpoint_path, compile=False)

        # Extract just the encoding sub model
        encoder_model = trained_siamese_model.get_layer('sequential_12')

        input = Input(shape=self.input_shape)
        x = encoder_model(input)
        model = Model(input, x)

        return model
