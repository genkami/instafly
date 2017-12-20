import abc

import keras.backend as K
import numpy as np
from keras.preprocessing.image import array_to_img
from keras.models import load_model
from keras.optimizers import SGD

from instafly import models
from instafly import config

class MomentumUpdator(object):
    def __init__(self, lr=0.01, gamma=0.01):
        self._lr = lr
        self._gamma = gamma
        self._v = None

    def update(self, param, loss, X):
        grads = K.gradients(loss, [param])[0]
        f = K.function([param], [loss, grads])

        if self._v is None:
            self._v = np.zeros(X.shape)

        loss_val, grads_val = f([X])
        self._v *= self._gamma
        self._v += self._lr * grads_val
        X -= self._v
        return loss_val

class Minimizer(object, metaclass=abc.ABCMeta):
    def __init__(self, updator=MomentumUpdator(), **kwargs):
        self._updator = updator

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

    def minimize_loss(self, num_loops=1000):
        X = self.create_initial_input()
        input = self.get_input()
        loss = self.get_loss()

        for i in range(num_loops):
            loss_val = self._updator.update(input, loss, X)
            print('%d: %f' % (i, loss_val))

        return X


class ImageVisualizer(Minimizer):
    def __init__(self, model, **kwargs):
        self._model = model
        super().__init__(**kwargs)

    def get_input(self):
        return self._model.input

    def get_loss(self):
        # Activationに通す前の出力
        return -self._model.layers[-2].output

    def create_initial_input(self):
        return np.random.random((1, config.IMG_WIDTH, config.IMG_HEIGHT, 3))

def main():
    model = load_model('../instafly-resources/weights.h5')
    visualizer = ImageVisualizer(
        model,
        updator=MomentumUpdator(lr=10)
    )
    img = visualizer.minimize_loss(num_loops=500)[0]
    img = array_to_img(img)
    img.save('output.png')

if __name__ == '__main__':
    main()
