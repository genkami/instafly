import argparse
from sklearn.model_selection import train_test_split

import instafly

def train(out_weight_path, epochs, csv_path, img_base_dir):
    model = instafly.models.create_model()
    X, Y = instafly.resources.load_images(csv_path, img_base_dir)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=instafly.config.TRAIN_SIZE
    )
    model.fit(
        X, Y,
        batch_size=instafly.config.BATCH_SIZE,
        epochs=epochs
    )
    model.save(out_weight_path)
    model.evaluate(
        X, Y,
        batch_size=instafly.config.BATCH_SIZE
    )

def main():
    argparser = argparse.ArgumentParser(
        description='モデルの学習を行う'
    )
    argparser.add_argument(
        '-n', metavar='NUM_EPOCHS', default=instafly.config.NUM_EPOCHS
    )
    argparser.add_argument(
        '-d', metavar='CSV_PATH', required=True
    )
    argparser.add_argument(
        '-i', metavar='IMG_BASE_DIR', required=True
    )
    argparser.add_argument(
        '-o', metavar='OUTPUT_PATH', required=True
    )
    args = argparser.parse_args()
    train(args.o, args.n, args.d, args.i)

if __name__ == '__main__':
    main()
