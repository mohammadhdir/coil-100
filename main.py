import gc
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard

import cresnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def get_label(file_path):
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    label = tf.strings.substr(file_name, 3, 2)

    if tf.strings.substr(label, 1, 1) == '_':
        first_num = tf.strings.substr(label, 0, 1)
        label = tf.strings.join(['0', first_num])

    label = int(label)
    return label


def process_image(image_file):
    image = tf.image.decode_png(image_file, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, [128, 128])
    image = image * 2 - 1
    return image


def load_image_label(image_file):
    label = get_label(image_file)
    image_file = tf.io.read_file(image_file)
    image = process_image(image_file)
    return image, label


# todo this method is not uset yet
def image_augmentation(image, label):
    image - tf.image.resize_with_crop_or_pad(image, 136, 136)
    image = tf.image.random_crop(image, [128, 128, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


path_to_zip = tf.keras.utils.get_file(
    'coil-100.zip',
    cache_subdir=os.path.abspath('.'),
    origin='http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip',
    extract=True,
)

dataset = tf.data.Dataset.list_files(os.path.abspath('.') + '/coil-100/*.png')
dataset_size = len(list(dataset))
batch_size = 32
epochs = 20
train_size = int(0.8 * dataset_size)

dataset = dataset.map(load_image_label, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(dataset_size)

train_data = dataset.take(train_size).batch(batch_size).prefetch(1)
test_data = dataset.skip(train_size).batch(batch_size).prefetch(1)

del dataset
gc.collect()

tbcb = TensorBoard(log_dir='log_dir', update_freq='batch', histogram_freq=1)

input_shape = (128, 128, 3)
img_input = keras.layers.Input(shape=input_shape)
opt = keras.optimizers.Adam()

model = cresnet.resnet56(img_input=img_input, classes=100)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=[tbcb])
