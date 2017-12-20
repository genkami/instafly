import abc
import argparse

import keras.backend as K
import numpy as np
from keras.preprocessing import image
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
    def __init__(self, model, initial_image=None, **kwargs):
        self._model = model
        self._initial_image = initial_image
        super().__init__(**kwargs)

    def get_input(self):
        return self._model.input

    def get_loss(self):
        # Activationに通す前の出力
        return -self._model.layers[-2].output

    def create_initial_input(self):
        if self._initial_image is None:
            return np.random.random((1, config.IMG_WIDTH, config.IMG_HEIGHT, 3))
        else:
            return self._initial_image

def main():
    parser = argparse.ArgumentParser(
        description='インスタ映えする画像を生成する'
    )
    parser.add_argument(
        '-i', metavar='INPUT_IMAGE', default=None
    )
    parser.add_argument(
        '-w', metavar='WEIGHT_FILE', required=True
    )
    parser.add_argument(
        '-o', metavar='OUTPUT_FILE', required=True
    )
    parser.add_argument(
        '-n', metavar='NUM_LOOPS', default=500
    )
    args = parser.parse_args()

    initial_image = None
    if args.i is not None:
        img = image.load_img(args.i, target_size=(config.IMG_WIDTH, config.IMG_HEIGHT))
        img = image.img_to_array(img).astype('float32')
        img /= 255
        initial_image = np.array([img])

    model = load_model(args.w)

    visualizer = ImageVisualizer(
        model,
        initial_image=initial_image,
        updator=MomentumUpdator(lr=10.0)
    )
    img = visualizer.minimize_loss(num_loops=int(args.n))[0]
    img = image.array_to_img(img)
    img.save(args.o)

if __name__ == '__main__':
    main()
