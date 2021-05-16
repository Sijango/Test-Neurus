from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model


def create_model(input_shape):
    inputs = Input(input_shape)

    conv1 = layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu')(inputs)
    max1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, (3, 3), input_shape=input_shape, activation='relu')(max1)
    max2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, (3, 3), input_shape=input_shape, activation='relu')(max2)
    max3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(64, (3, 3), input_shape=input_shape, activation='relu')(max3)
    max4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(64, (3, 3), input_shape=input_shape, activation='relu')(max4)
    max5 = layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    flat = layers.Flatten()(max5)

    y_class = layers.Dense(240, activation='relu')(flat)
    y_class = layers.Dense(240, activation='relu')(y_class)
    y_class = layers.Dropout(0.5)(y_class)
    out_class = layers.Dense(1, activation='sigmoid', name='class')(y_class)

    y_xmin1 = layers.Dense(620, activation='relu')(flat)
    y_xmin2 = layers.Dense(620, activation='relu')(y_xmin1)
    y_xmin3 = layers.Dense(620, activation='relu')(y_xmin2)
    y_xmin4 = layers.Dropout(0.5)(y_xmin3)
    out_xmin = layers.Dense(1, activation='linear', name='xmin')(y_xmin4)

    y_ymin1 = layers.Dense(620, activation='relu')(flat)
    y_ymin2 = layers.Dense(620, activation='relu')(y_ymin1)
    y_ymin3 = layers.Dense(620, activation='relu')(y_ymin2)
    y_ymin4 = layers.Dropout(0.5)(y_ymin3)
    out_ymin = layers.Dense(1, activation='linear', name='ymin')(y_ymin4)

    y_xmax1 = layers.Dense(620, activation='relu')(flat)
    y_xmax2 = layers.Dense(620, activation='relu')(y_xmax1)
    y_xmax3 = layers.Dense(620, activation='relu')(y_xmax2)
    y_xmax4 = layers.Dropout(0.5)(y_xmax3)
    out_xmax = layers.Dense(1, activation='linear', name='xmax')(y_xmax4)

    y_ymax1 = layers.Dense(620, activation='relu')(flat)
    y_ymax2 = layers.Dense(620, activation='relu')(y_ymax1)
    y_ymax3 = layers.Dense(620, activation='relu')(y_ymax2)
    y_ymax4 = layers.Dropout(0.5)(y_ymax3)
    out_ymax = layers.Dense(1, activation='linear', name='ymax')(y_ymax4)

    x_y_list = layers.concatenate([out_xmin, out_ymin, out_xmax, out_ymax])
    x_y_list = layers.Dense(4, activation='relu', name='out')(x_y_list)

    return Model(inputs, [out_class, x_y_list])


if __name__ == '__main__':

    model = create_model((220, 220, 3))
    plot_model(model, 'model1.png')
    print(model.summary())
