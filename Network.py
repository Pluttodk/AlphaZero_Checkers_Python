# import needed classes
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization, \
    Activation, LeakyReLU, add
import numpy as np
from Abstract_Game import *

from tensorflow.keras.optimizers import Adam
from math import ceil
import os

from Config import Config


def conv_layer(x, filters, kernel_size):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=keras.regularizers.l2(Config.REG_VAL)
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)

    return x


def residual_layer(input_block, filters, kernel_size):
    x = conv_layer(input_block, filters, kernel_size)

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        data_format="channels_first",
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=keras.regularizers.l2(Config.REG_VAL)
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)
    return x


def value_head(x):
    x = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=keras.regularizers.l2(Config.REG_VAL),
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(
        20,
        use_bias=False,
        activation='linear',
    )(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1,
                           activation=keras.activations.tanh,
                           name="value_head"
                           )(x)

    return x


def policy_head(x, output_dim):
    x = Conv2D(
        filters=2,
        kernel_size=(1, 1),
        data_format="channels_first",
        padding="same",
        use_bias=False,
        activation='linear',
        kernel_regularizer=keras.regularizers.l2(0.0001)
    )(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dense(
        output_dim,
        use_bias=False,
        activation=keras.activations.softmax,
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name="policy_head"
    )(x)

    return x


def softmax_cross_entropy_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=p)

    return loss


class Network:

    def __init__(self, shape=(0,0), output_dim=(0,0), network_path=None):
        if network_path:
            self.model = tf.keras.models.load_model(network_path,
                                                  custom_objects={'softmax_cross_entropy_logits':
                                                                      softmax_cross_entropy_logits})
        else:
            main_input = keras.Input(shape=shape, name='main_input')
            x = conv_layer(main_input, Config.HIDDEN_CNN_LAYERS[0]['filters'], Config.HIDDEN_CNN_LAYERS[0]['kernel_size'])

            for h in Config.HIDDEN_CNN_LAYERS[1:]:
                x = residual_layer(x, h['filters'], h['kernel_size'])

            vh = value_head(x)
            ph = policy_head(x, output_dim)

            model = keras.Model(inputs=[main_input], outputs=[vh, ph])
            model.compile(loss={"value_head": 'mean_squared_error', "policy_head": softmax_cross_entropy_logits},
                          optimizer=keras.optimizers.SGD(lr=Config.LEARNING_RATIO,
                                                         decay=Config.DECAY,
                                                         momentum=Config.MOMENTUM),
                          loss_weights={"value_head": 0.5, "policy_head": 0.5}
                          )
            self.model = model

    def evaluate(self, image, size, turn):
        """

        evaluate the network based on a value
        :return:
        """
        outcome = np.zeros((2, size, size))
        for y in range(len(image)):
            for x in range(len(image[0])):
                opponent = 1 if turn - 1 else 2
                if image[y][x] == turn:
                    outcome[0][y][x] = 1
                elif image[y][x] == opponent:
                    outcome[1][y][x] = 1

        outcome = outcome.reshape((-1, 2, size, size))
        res = self.model.predict(outcome)
        output = res[0][0][0], res[1][0]
        return output

    def evaluate_multiple(self, image):
        res = self.model.predict(image)
        output = res[0], res[1]
        return output

    def save_network(self, test_version):
        path = f"Output/{Config.NETWORK_NAME}{test_version}.h5"
        self.model.save(path)

    def get_weights(self):
        """
        Returns the weights
        :return:
        """
        return self.model.get_weights()
