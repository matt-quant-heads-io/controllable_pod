from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf


def get_conditional_cnn_model(obs_size, action_dim, num_conditions):
    inputs = [Input(shape=(obs_size, obs_size, action_dim)), Input(shape=(num_conditions,))]

    x = Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, action_dim), padding='SAME')(inputs[0])
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='SAME')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='SAME')(x)
    x = Flatten()(x)
    x = Concatenate()([x, inputs[1]])
    x = Dense(128)(x)

    output = Dense(action_dim, activation='softmax')(x)

    conditional_cnn_model = Model(input=inputs, output=output)

    conditional_cnn_model.compile(loss='categorical_crossentropy', optimizer=SGD(),
                       metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return conditional_cnn_model