import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

TXT_DIR = '../images/test'
IMAGE_DIR = '../images/test'
LABELS_PATH = '../annotations/label_map.pbtxt'
OUTPUT_PATH = '../annotations/test.record'
CSV_PATH = '../annotations/test.csv'

label_map = label_map_util.load_labelmap(LABELS_PATH)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def txt_to_csv(path):
    txt_list = []
    for txt_file in glob.glob(path + '/*.txt'):
        full_name = os.path.basename(txt_file)
        name = os.path.splitext(full_name)[0]

        image = Image.open(path + '/' + name + '.jpg')
        width, height = image.size

        with open(txt_file, 'r', encoding='utf-8') as f:
            tmp = f.readline().split(' ')
            if int(tmp[0]) == 1:
                tmp_class = 'cat'.encode()
            else:
                tmp_class = 'dog'.encode()

            value = (name + '.jpg',
                     int(width),
                     int(height),
                     tmp_class,
                     int(tmp[1]),
                     int(tmp[3]),
                     int(tmp[2]),
                     int(tmp[4])
                     )

            txt_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'xmax', 'ymin', 'ymax']

    txt_df = pd.DataFrame(txt_list, columns=column_name)
    txt_df.to_csv("dataset.csv")

    return txt_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)

    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf-8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'])
        classes.append(class_text_to_int(row['class'].decode()))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)
    path = os.path.join(IMAGE_DIR)
    examples = txt_to_csv(TXT_DIR)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(OUTPUT_PATH))
    if CSV_PATH is not None:
        examples.to_csv(CSV_PATH, index=None)
        print('Successfully created the CSV file: {}'.format(CSV_PATH))


if __name__ == '__main__':
    tf.app.run()
