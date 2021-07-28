import tensorflow as tf
from tensorflow.keras import layers


def unetModel():
    # declaring the input layer
    # Input layer expects an RGB image, in the original paper the network consisted of only one channel.
    inputs = layers.Input(shape=(448, 448, 3))
    # first part of the U - contracting part
    c0 = layers.Conv2D(64, activation='relu', padding="same", kernel_size=3)(inputs)
    c1 = layers.Conv2D(64, activation='relu', padding="same", kernel_size=3)(c0)  # This layer for concatenating in the expansive part
    c2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)

    c3 = layers.Conv2D(128, activation='relu', padding="same", kernel_size=3)(c2)
    c4 = layers.Conv2D(128, activation='relu', padding="same", kernel_size=3)(c3)  # This layer for concatenating in the expansive part
    c5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c4)

    c6 = layers.Conv2D(256, activation='relu', padding="same", kernel_size=3)(c5)
    c7 = layers.Conv2D(256, activation='relu', padding="same", kernel_size=3)(c6)  # This layer for concatenating in the expansive part
    c8 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c7)

    c9 = layers.Conv2D(512, activation='relu', padding="same", kernel_size=3)(c8)
    c10 = layers.Conv2D(512, activation='relu', padding="same", kernel_size=3)(c9)  # This layer for concatenating in the expansive part
    c11 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c10)

    c12 = layers.Conv2D(1024, activation='relu', padding="same", kernel_size=3)(c11)
    c13 = layers.Conv2D(1024, activation='relu', padding='same', kernel_size=3)(c12)

    # We will now start the second part of the U - expansive part
    t01 = layers.Conv2DTranspose(512, kernel_size=2, strides=(2, 2), activation='relu', padding='same')(c13)
    concat01 = layers.concatenate([t01, c10], axis=-1)

    c14 = layers.Conv2D(512, activation='relu', kernel_size=3, padding='same')(concat01)
    c15 = layers.Conv2D(512, activation='relu', kernel_size=3, padding='same')(c14)

    t02 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu', padding='same')(c15)
    concat02 = layers.concatenate([t02, c7], axis=-1)

    c16 = layers.Conv2D(256, activation='relu', kernel_size=3, padding='same')(concat02)
    c17 = layers.Conv2D(256, activation='relu', kernel_size=3, padding='same')(c16)

    t03 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu', padding='same')(c17)
    concat03 = layers.concatenate([t03, c4], axis=-1)

    c18 = layers.Conv2D(128, activation='relu', kernel_size=3, padding='same')(concat03)
    c19 = layers.Conv2D(128, activation='relu', kernel_size=3, padding='same')(c18)

    t04 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), activation='relu', padding='same')(c19)
    concat04 = layers.concatenate([t04, c1], axis=-1)

    c20 = layers.Conv2D(64, activation='relu', kernel_size=3, padding='same')(concat04)
    c21 = layers.Conv2D(64, activation='relu', kernel_size=3, padding='same')(c20)

    # This is based on our dataset. The output channels are 3, think of it as each pixel will be classified
    # into three classes, but I have written 4 here, as I do padding with 0, so we end up have four classes.
    outputs = layers.Conv2D(4, kernel_size=1)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")
    return model
