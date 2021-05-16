import os

from keras import losses
from tensorflow import keras
import tensorflow as tf

from clear.data import get_data
from clear.metric import custom_loss, mean_iou
from clear.model import create_model

if __name__ == '__main__':
    path_train_jpg = '../cats_dogs_dataset/train/*.jpg'
    path_train_txt = '../cats_dogs_dataset/train/*.txt'

    path_valid_jpg = '../cats_dogs_dataset/valid/*.jpg'
    path_valid_txt = '../cats_dogs_dataset/valid/*.txt'

    x_train, y_train, x_test, y_test, x_valid, y_valid = get_data(
        path_train_jpg,
        path_train_txt,
        path_valid_jpg,
        path_valid_txt)

    input_shape = (220, 220, 3)

    if os.path.exists('model.h5'):
        model = tf.keras.models.load_model('model.h5', custom_objects={'custom_loss': custom_loss, 'mean_iou': mean_iou})
    else:
        model = create_model(input_shape)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                      loss={
                          'class': losses.BinaryCrossentropy(),
                          'out': custom_loss
                      },
                      metrics={
                          'class': 'accuracy',
                          'out': mean_iou
                      })

        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=32, epochs=10)
        model.save('model.h5')

    loss, class_loss, out_loss, test_acc, iou = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy: ', test_acc, ' iou: ', iou)

    predictions = model.predict(x_test)

    print(f'\n Predictions 1:\n {predictions[0]}')
