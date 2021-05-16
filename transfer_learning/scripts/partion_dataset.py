import os
import re
from shutil import copy
import argparse
import math
import random


def iterate_dir(source='../../cats_dogs_dataset/train', dest='../images', ratio=0.1, copy_txt=True):
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')
    train_dir = train_dir.replace('\\', '/')
    test_dir = test_dir.replace('\\', '/')
    print(f'[dir] OK: train: {train_dir}, test: {test_dir}')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print('[create train dir] OK')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print('[create test dir] OK')
    print('[create dir] NOT')

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)
    print(f'[nums] OK: images = {num_images}, test_images = {num_test_images}')

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]

        src = os.path.join(source, filename)
        src = src.replace("\\", "/")
        copy(src, test_dir)
        print(f'[copy test] OK {test_dir}')

        if copy_txt:
            txt_filename = os.path.splitext(filename)[0]+'.txt'

            src_txt = os.path.join(source, txt_filename)
            src_txt = src_txt.replace("\\", "/")
            copy(src_txt, test_dir)
            print(f'[copy test] OK {test_dir}')

        images.pop(idx)

    for filename in images:
        src = os.path.join(source, filename)
        src = src.replace("\\", "/")
        copy(src, train_dir)
        print(f'[copy train] OK {train_dir}')

        if copy_txt:
            txt_filename = os.path.splitext(filename)[0]+'.txt'
            src_txt = os.path.join(source, txt_filename)
            src_txt = src_txt.replace("\\", "/")
            copy(src_txt, train_dir)
            print(f'[copy train] OK {train_dir}')


if __name__ == '__main__':
    iterate_dir()
