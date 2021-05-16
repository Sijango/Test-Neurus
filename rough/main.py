import os

from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import losses
from keras.utils.generic_utils import get_custom_objects

from img_or_txt_func import get_image_data, get_txt, get_txt_index
from mIOU_func import iou_metric, custom_loss, mean_iou
from model_func import create_model, create_model_v3

tf.config.run_functions_eagerly(True)

if __name__ == '__main__':
    # x_train = get_image_data('cats_dogs_dataset/train/*.jpg')
    # y_train = get_txt('cats_dogs_dataset/train/*.txt')
    #
    # x_test = get_image_data('cats_dogs_dataset/valid/*.jpg')
    # y_test = get_txt('cats_dogs_dataset/valid/*.txt')

    images = get_image_data('../cats_dogs_dataset/train/*.jpg')
    txt = get_txt('../cats_dogs_dataset/train/*.txt')
    txt_class = get_txt_index('../cats_dogs_dataset/train/*.txt', 0)
    txt_xmin = get_txt_index('../cats_dogs_dataset/train/*.txt', 1)
    txt_ymin = get_txt_index('../cats_dogs_dataset/train/*.txt', 2)
    txt_xmax = get_txt_index('../cats_dogs_dataset/train/*.txt', 3)
    txt_ymax = get_txt_index('../cats_dogs_dataset/train/*.txt', 4)

    # images_valid = get_image_data('cats_dogs_dataset/valid/*.jpg')
    # txt_valid = get_txt('cats_dogs_dataset/valid/*.txt')

    images_valid = get_image_data('../cats_dogs_dataset/valid/*.jpg')
    txt_class_valid = get_txt_index('../cats_dogs_dataset/valid/*.txt', 0)
    txt_xmin_valid = get_txt_index('../cats_dogs_dataset/valid/*.txt', 1)
    txt_ymin_valid = get_txt_index('../cats_dogs_dataset/valid/*.txt', 2)
    txt_xmax_valid = get_txt_index('../cats_dogs_dataset/valid/*.txt', 3)
    txt_ymax_valid = get_txt_index('../cats_dogs_dataset/valid/*.txt', 4)

    # X = np.ndarray(images)
    # Y = np.concatenate([txt_class, txt_xmin, txt_ymin, txt_xmax, txt_ymax], axis=5)
    Y = np.concatenate([txt], axis=1)
    x_train, x_test, _, _ = train_test_split(images, Y, test_size=0.1)

    _, _, y_train_class, y_test_class = train_test_split(images, txt_class, test_size=0.1)
    _, _, y_train_xmin, y_test_xmin = train_test_split(images, txt_xmin, test_size=0.1)
    _, _, y_train_ymin, y_test_ymin = train_test_split(images, txt_ymin, test_size=0.1)
    _, _, y_train_xmax, y_test_xmax = train_test_split(images, txt_xmax, test_size=0.1)
    _, _, y_train_ymax, y_test_ymax = train_test_split(images, txt_ymax, test_size=0.1)

    x_train = np.array(x_train)
    # y_train = np.array(y_train)

    y_train_class = np.array(y_train_class)
    y_train_xmin = np.array(y_train_xmin)
    y_train_ymin = np.array(y_train_ymin)
    y_train_xmax = np.array(y_train_xmax)
    y_train_ymax = np.array(y_train_ymax)

    # y_train_class = keras.utils.to_categorical(y_train_class, 3)
    # y_train_xmin = keras.utils.to_categorical(y_train_xmin, 1000)
    # y_train_ymin = keras.utils.to_categorical(y_train_ymin, 1000)
    # y_train_xmax = keras.utils.to_categorical(y_train_xmax, 1000)
    # y_train_ymax = keras.utils.to_categorical(y_train_ymax, 1000)

    y_test_class = np.array(y_test_class)
    y_test_xmin = np.array(y_test_xmin)
    y_test_ymin = np.array(y_test_ymin)
    y_test_xmax = np.array(y_test_xmax)
    y_test_ymax = np.array(y_test_ymax)

    # y_test_class = keras.utils.to_categorical(y_test_class, 3)
    # y_test_xmin = keras.utils.to_categorical(y_test_xmin, 1000)
    # y_test_ymin = keras.utils.to_categorical(y_test_ymin, 1000)
    # y_test_xmax = keras.utils.to_categorical(y_test_xmax, 1000)
    # y_test_ymax = keras.utils.to_categorical(y_test_ymax, 1000)

    images_valid = np.array(images_valid)
    txt_class_valid = np.array(txt_class_valid)
    txt_xmin_valid = np.array(txt_xmin_valid)
    txt_ymin_valid = np.array(txt_ymin_valid)
    txt_xmax_valid = np.array(txt_xmax_valid)
    txt_ymax_valid = np.array(txt_ymax_valid)

    y_train_cat = {'class': y_train_class,
                   'out': [y_train_xmin, y_train_ymin, y_train_xmax, y_train_ymax]}
                   # 'xmin': y_train_xmin,
                   # 'ymin': y_train_ymin,
                   # 'xmax': y_train_xmax,
                   # 'ymax': y_train_ymax}

    y_test_cat = {'class': y_test_class,
                  'out': [y_test_xmin, y_test_ymin, y_test_xmax, y_test_ymax]}
                   # 'xmin': y_test_xmin,
                   # 'ymin': y_test_ymin,
                   # 'xmax': y_test_xmax,
                   # 'ymax': y_test_ymax}

    valid_cat = {'class': txt_class_valid,
                  'out': [txt_xmin_valid, txt_ymin_valid, txt_xmax_valid, txt_ymax_valid]}

    x_test = np.array(x_test)
    # y_test = np.array(y_test)

    # x_valid = np.array(x_valid)
    # y_valid = np.array(y_valid)

    # print(y_train.shape())
    print(len(x_test))
    print(len(y_train_xmin))
    # print(len(y_test))

    # y_train_cat = keras.utils.to_categorical(y_train, 5)
    # y_test_cat = keras.utils.to_categorical(y_test, 5)
    # x_train = np.expand_dims(x_train, axis=3)
    # x_test = np.expand_dims(x_test, axis=3)

    # print('\n txt: ', txt[0])
    # print('\n y_train: ', y_train[0])

    # print(x_train.shape)
    input_shape = (220, 220, 3)
    if os.path.exists('model.h5'):
        model = tf.keras.models.load_model('model.h5', custom_objects={'custom_loss': custom_loss, 'mean_iou': mean_iou})
    else:
        model = create_model_v3(input_shape)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                      loss={
                          'class': losses.BinaryCrossentropy(),
                          'out': custom_loss
                          # 'xmin': losses.MeanSquaredError(),
                          # 'ymin': losses.MeanSquaredError(),
                          # 'xmax': losses.MeanSquaredError(),
                          # 'ymax': losses.MeanSquaredError()
                      },
                      metrics={
                          'class': 'accuracy',
                          'out': mean_iou
                          # ['xmin', 'ymin', 'xmax', 'ymax']: iou_metric
                      } # , iou_metric(y_pred=['xmin', 'ymin', 'xmax', 'ymax'])]
                      # metrics={
                      #     'class': 'accuracy',
                      #     'xmin': 'accuracy',
                      #     'ymin': 'accuracy',
                      #     'xmax': 'accuracy',
                      #     'ymax': 'accuracy'}
                      )

        # datagen = ImageDataGenerator(rescale=1. / 255)
        #
        # train_generator = datagen.flow_from_directory(
        #     'cats_dogs_dataset/train',
        #     target_size=(220, 220),
        #     batch_size=32,
        #     class_mode='categorical'
        # )
        #
        # valid_generator = datagen.flow_from_directory(
        #     'cats_dogs_dataset/valid',
        #     target_size=(220, 220),
        #     batch_size=32,
        #     class_mode='categorical'
        # )

        # model.fit_generator(
        #     train_generator,
        #     epochs=10,
        #     validation_data=valid_generator
        # )

        model.fit(x_train, y_train_cat, validation_data=(images_valid, valid_cat), batch_size=32, epochs=10)
        model.save('model.h5')

    loss, class_loss, out_loss, test_acc, iou = model.evaluate(x_test, y_test_cat, verbose=2)
    print('\nTest accuracy: ', test_acc, ' iou: ', iou)

    predictions = model.predict(x_test)

    print(f'\n Predictions 1:\n {predictions[0]}')
