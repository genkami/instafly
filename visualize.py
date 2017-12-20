import abc

import keras.backend as K
import numpy as np
from keras.preprocessing.image import array_to_img
from keras.models import load_model

from instafly import models
from instafly import config

class Minimizer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_input(self):
        '''
        ニューラルネットの入力部分を返す。
        戻り値は任意のtensor
        '''
        ...

    @abc.abstractmethod
    def get_loss(self):
        '''
        最小化したいニューラルネットの計算結果を返す。
        戻り値はスカラー
        '''
        ...

    @abc.abstractmethod
    def create_initial_input(self):
        '''
        ニューラルネットの入力の初期値を生成する。
        戻り値はnumpy行列
        '''
        ...

    def minimize_loss(self, lr=1, num_loops=5000):
        X = self.create_initial_input()
        input = self.get_input()
        loss = self.get_loss()
        grads = lr * K.gradients(loss, input)[0]
        func = K.function([input], [loss, grads])

        for i in range(num_loops):
            loss_value, grads_value = func([X])
            print('%d: %f' % (i, loss_value))
            X -= grads_value

        return X


class ImageVisualizer(Minimizer):
    def __init__(self, model):
        self._model = model

    def get_input(self):
        return self._model.input

    def get_loss(self):
        # Activationに通す前の出力
        return -self._model.layers[-2].output

    def create_initial_input(self):
        return np.random.random((1, config.IMG_WIDTH, config.IMG_HEIGHT, 3))

def main():
    model = load_model('../instafly-resources/weights.h5')
    visualizer = ImageVisualizer(model)
    img = visualizer.minimize_loss()[0]
    img = array_to_img(img)
    img.save('output.png')

if __name__ == '__main__':
    main()
