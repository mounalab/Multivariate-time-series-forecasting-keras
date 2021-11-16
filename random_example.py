
import random
import numpy as np


import keras
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.layers.core import Dense, Activation, Dropout, Flatten

from siamesenet import SiameseNetwork
from distances_utils import contrastive_loss


# Training parameters
batch_size = 128
n_epochs = 999999
checkpoint_path = "./checkpoint"
log_path = "./log"
validation_split = 0.2

# Siamese network parameters
n_output = 20 # the size of embeddings (output of the encoder model)
learning_rate = 1e-2


def create_encoder_model(input_shape):
    """ Encoder network to be shared

    :param input_shape: the shape of the input layer
    :type int
    :return: the model structure
    :rtype: Keras Sequential model
    """

    # define model
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, name='fc1')) #, activation='relu'
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU())
    model.add(Dropout(0.1))

    model.add(Dense(128, name='fc2'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(n_output, activation='relu', name='fc4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))

    #kernel_initializer='random_uniform'

    return model


""" Randomly generating dummy training data for the Siamese network
"""

n_examples = 7500
n_features = 25

x_train_1 = np.random.random((n_examples, n_features))
x_train_2 = np.random.random((n_examples, n_features))
y_train = np.random.random((n_examples))

x_test = np.random.random((n_examples, n_features))

input_shape = n_features


""" Building the Siamese Network
"""

encoder_model = create_encoder_model(input_shape)
siamese_model = SiameseNetwork(encoder_model)



""" Compile
"""

# Define the optimizer and compile the model
rms = RMSprop()#learning_rate=0.001)
sgd = SGD(lr=learning_rate, momentum=0.9, decay=0, nesterov=True)
adam = Adam()
nadam = Nadam(lr=learning_rate)

siamese_model.compile(loss=contrastive_loss, optimizer=adam, metrics=['mae'])


""" Training
"""

siamese_model.fit(x_train_1, x_train_2, y_train, n_epochs=n_epochs, batch_size=batch_size, validation_split=validation_split, checkpoint_path=checkpoint_path, log_path=log_path)


""" Using the trained encoder
Extracting new feature embeddings from test data
"""

model = siamese_model.restore(encoder_model=encoder_model, checkpoint_path=checkpoint_path)
print(model.summary())
new_embeddings = model.predict(x_test)
