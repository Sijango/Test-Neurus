import numpy as np
import glob
from PIL import Image


#  ФУНКЦИЯ ДЛЯ ПРЕОБРАЗОВАНИЯ ФОТОГРАФИЙ ПОД НУЖНЫЙ ФОРМАТ
def get_images(path):
    image_paths = glob.glob(path)

    for image_file in image_paths:
        image = Image.open(image_file).resize((220, 220))
        image = np.asarray(image.convert('RGB')) / 255.0

        yield image


#  ФУНКЦИЯ ДЛЯ ПРЕОБРАЗОВАНИЯ ФАЙЛОВ К ФОТО ПОД НУЖНЫЙ ФОРМАТ
def get_txt(path):
    txt_data = []
    txt_path = glob.glob(path)

    for txt_file in txt_path:
        with open(txt_file, 'r', encoding='utf-8') as f:
            tmp = f.readline().split(' ')
            # tmp = int(tmp[0])
            bndbox = [None] * 5
            bndbox[0] = int(tmp[0])
            bndbox[1] = int(tmp[1])
            bndbox[2] = int(tmp[2])
            bndbox[3] = int(tmp[3])
            bndbox[4] = int(tmp[4])

            txt_data.append(bndbox)
            # txt_data.append(tmp)

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


def get_image_data(path):
    images_data = []

    images = get_images(path)
    for img in images:
        images_data.append(img)

    return images_data


if __name__ == '__main__':
    images_data = []

    images = get_images('cats_dogs_dataset/train/*.jpg')
    for img in images:
        images_data.append(img)

    print(len(images_data))

    # path = 'data/image_data.txt'
    # save_data_img(path, images)

    # for image in images:  # Работает оч долго, но оно работает )
    #     print(image)
    #
    # txt = get_txt('cats_dogs_dataset/train/*.txt')
    #
    # for item in txt:
    #     print(item)
