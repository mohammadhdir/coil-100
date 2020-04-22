import gc
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def get_label(file_path):
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    label = tf.strings.substr(file_name, 3, 2)

    if tf.strings.substr(label, 1, 1) == '_':
        first_num = tf.strings.substr(label, 0, 1)
        label = tf.strings.join(['0', first_num])

    return int(label)


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


# todo this method is not used
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

input_shape = (128, 128, 3)
tbcb = TensorBoard(log_dir='log_dir', update_freq='batch', histogram_freq=1)
opt = keras.optimizers.Adam()

model = keras.models.Sequential()
model.add(layers.Input(input_shape, name='input'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', name='conv-1'))
model.add(layers.MaxPooling2D((2, 2), name='maxpool-1'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv-2'))
model.add(layers.MaxPooling2D((2, 2), name='maxpool-2'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv-3'))

model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(512, activation='relu', name='dense-1'))
model.add(layers.Dense(128, activation='relu', name='dense-2'))
model.add(layers.Dense(100, activation='softmax', name='out'))

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=[tbcb])
