import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split


#  ФУНКЦИЯ ДЛЯ ПРЕОБРАЗОВАНИЯ ФОТОГРАФИЙ ПОД НУЖНЫЙ ФОРМАТ
def get_images(path):
    image_paths = glob.glob(path)

    for image_file in image_paths:
        image = Image.open(image_file).resize((220, 220))
        image = np.asarray(image.convert('RGB')) / 255.0

        yield image


def get_image_data(path):
    images_data = []

    images = get_images(path)
    for img in images:
        images_data.append(img)

    return images_data


#  ФУНКЦИЯ ДЛЯ ПРЕОБРАЗОВАНИЯ ФАЙЛОВ К ФОТО ПОД НУЖНЫЙ ФОРМАТ
def get_txt(path):
    txt_data = []
    txt_path = glob.glob(path)

    for txt_file in txt_path:
        with open(txt_file, 'r', encoding='utf-8') as f:
            tmp = f.readline().split(' ')

            bndbox = [None] * 5
            bndbox[0] = int(tmp[0])
            bndbox[1] = int(tmp[1])
            bndbox[2] = int(tmp[2])
            bndbox[3] = int(tmp[3])
            bndbox[4] = int(tmp[4])

            txt_data.append(bndbox)

    return txt_data


def get_txt_index(path, index):
    txt_data = []
    txt_path = glob.glob(path)

    for txt_file in txt_path:
        with open(txt_file, 'r', encoding='utf-8') as f:
            tmp = f.readline().split(' ')
            tmp = int(tmp[index])
            txt_data.append(tmp)

    return txt_data


# Получение всех данных
def get_data(path_train_jpg, path_train_txt):
    images = get_image_data(path_train_jpg)
    txt = get_txt(path_train_txt)

    txt_class = get_txt_index(path_train_txt, 0)
    txt_xmin = get_txt_index(path_train_txt, 1)
    txt_ymin = get_txt_index(path_train_txt, 2)
    txt_xmax = get_txt_index(path_train_txt, 3)
    txt_ymax = get_txt_index(path_train_txt, 4)

    Y = np.concatenate([txt], axis=1)

    x_train, x_test, _, _ = train_test_split(images, Y, test_size=0.1)

    _, _, y_train_class, y_test_class = train_test_split(images, txt_class, test_size=0.1)
    _, _, y_train_xmin, y_test_xmin = train_test_split(images, txt_xmin, test_size=0.1)
    _, _, y_train_ymin, y_test_ymin = train_test_split(images, txt_ymin, test_size=0.1)
    _, _, y_train_xmax, y_test_xmax = train_test_split(images, txt_xmax, test_size=0.1)
    _, _, y_train_ymax, y_test_ymax = train_test_split(images, txt_ymax, test_size=0.1)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    y_train_class = np.array(y_train_class)
    y_train_xmin = np.array(y_train_xmin)
    y_train_ymin = np.array(y_train_ymin)
    y_train_xmax = np.array(y_train_xmax)
    y_train_ymax = np.array(y_train_ymax)

    y_test_class = np.array(y_test_class)
    y_test_xmin = np.array(y_test_xmin)
    y_test_ymin = np.array(y_test_ymin)
    y_test_xmax = np.array(y_test_xmax)
    y_test_ymax = np.array(y_test_ymax)

    y_train_cat = {'class': y_train_class,
                   'xmin': y_train_xmin,
                   'ymin': y_train_ymin,
                   'xmax': y_train_xmax,
                   'ymax': y_train_ymax}

    y_test_cat = {'class': y_test_class,
                   'xmin': y_test_xmin,
                   'ymin': y_test_ymin,
                   'xmax': y_test_xmax,
                   'ymax': y_test_ymax}

    return x_train, y_train_cat, x_test, y_test_cat


if __name__ == '__main__':
    images_data = []

    images = get_images('cats_dogs_dataset/train/*.jpg')
    for img in images:
        images_data.append(img)

    print(len(images_data))
