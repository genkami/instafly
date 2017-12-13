from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.vgg16 import VGG16

from . import config

NUM_HIDDEN = 16
NUM_CHANNELS = 3
DEFAULT_ACTIVATION='relu'

def create_model():
    model = Sequential()
    vgg16 = VGG16(
        input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, NUM_CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    for layer in vgg16.layers[:-1]:
        # 最も出力に近い層のみ学習させる
        layer.trainable = False
    model.add(vgg16)
    model.add(Activation(DEFAULT_ACTIVATION))
    model.add(Flatten())
    model.add(Dense(NUM_HIDDEN))
    model.add(Activation(DEFAULT_ACTIVATION))
    model.add(Dense(1))
    model.add(Activation('relu')) # 出力は0以上なので

    model.compile(
        loss='mse',
        optimizer='sgd'
    )

    return model
