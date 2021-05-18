import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from VGG16.data import get_data, get_txt_index, get_image_data

if __name__ == '__main__':
    path_train_img = '../../cats_dogs_dataset/train/*.jpg'
    path_train_txt = '../../cats_dogs_dataset/train/*.txt'

    images = get_image_data(path_train_img)
    txt = get_txt_index(path_train_txt, 0)

    x_train, x_test, y_train, y_test = train_test_split(images, txt, test_size=0.1)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model_VGG16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(220, 220, 3))

    model_VGG16.trainable = False

    model = keras.Sequential()
    model.add(model_VGG16)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-5),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy', test_acc)
