from keras import losses
from tensorflow import keras
import tensorflow as tf
import numpy as np

from clear.data import get_data
from clear.model import create_model

if __name__ == '__main__':
    path_train_jpg = '../cats_dogs_dataset/train/*.jpg'
    path_train_txt = '../cats_dogs_dataset/train/*.txt'

    x_train, y_train, x_test, y_test = get_data(path_train_jpg)

    input_shape = (220, 220, 3)

    model = create_model(input_shape)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss={
                      'class': losses.SparseCategoricalCrossentropy(),
                      'xmin': losses.SparseCategoricalCrossentropy(from_logits=True),
                      'ymin': losses.SparseCategoricalCrossentropy(from_logits=True),
                      'xmax': losses.SparseCategoricalCrossentropy(from_logits=True),
                      'ymax': losses.SparseCategoricalCrossentropy(from_logits=True)
                  },
                  metrics={
                      'class': 'accuracy',
                      'xmin': 'accuracy',
                      'ymin': 'accuracy',
                      'xmax': 'accuracy',
                      'ymax': 'accuracy'
                  })

    model.fit(x_train, y_train, batch_size=32, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy', test_acc)