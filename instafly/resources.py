import os
import csv
from keras.preprocessing import image
import numpy as np

from . import config

class Metadata(object):
    def __init__(self, path, like_rate):
        self.path = path
        self.like_rate = like_rate

def flow_metadata(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        headers = None
        for row in reader:
            if headers is None:
                # 最初の行はヘッダ
                headers = row
                continue
            row_dict = { k: v for k, v in zip(headers, row) }
            yield Metadata(
                path=row_dict['path'],
                like_rate=row_dict['like_rate']
            )

def load_images(csv_path, img_base_dir):
    X = []
    Y = []
    for meta in flow_metadata(csv_path):
        path = os.path.join(img_base_dir, meta.path)
        img = image.load_img(path, target_size=(config.IMG_WIDTH, config.IMG_HEIGHT))
        img = image.img_to_array(img).astype('float32')
        img /= 255
        X.append(img)
        Y.append(meta.like_rate)
    return np.array(X), np.array(Y).astype('float32')
