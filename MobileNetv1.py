import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, AveragePooling2D, Activation, Dense
from tensorflow.keras import Input
import tensorflow_datasets as tfds
import os

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR'])
strategy = tf.distribute.TPUStrategy(resolver)
(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    'Cifar10',
    split = ['train', 'test'],
    batch_size = -1,
    as_supervised = True,
))
y_train = tf.one_hot(y_train, depth = 10)
y_test = tf.one_hot(y_test, depth = 10)

def mobilenetv1(x, alph = 1):
    def dw(x, dw_pad, conv_f, conv_st):
        x = DepthwiseConv2D(kernel_size = (3,3), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters = conv_f, kernel_size = (1,1), strides = conv_st, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    x = Conv2D(filters = int(32 * alph), kernel_size = (3,3), strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = dw(x, 'same', int(64 * alph), 1)
    x = dw(x, 'valid', int(128 * alph), 2)
    x = dw(x, 'same', int(128 * alph), 1)
    x = dw(x, 'same', int(256 * alph), 2)
    x = dw(x, 'same', int(256 * alph), 1)
    x = dw(x, 'valid', int(512 * alph), 2)
    for i in range(5):
        x = dw(x, 'same', int(512 * alph), 1)
    x = dw(x, 'valid', int(1024 * alph), 2)
    x = dw(x, 'same', int(1024 * alph), 1)
    return x

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(30, 128)
checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auth')
earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10)
reduceLR = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3,)

with strategy.scope():
    inputs = Input(shape = (32, 32, 3), dtype = np.float32)
    x = mobilenetv1(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation = 'softmax')(x)
    model = tf.kears.models.Model(inputs, outputs)
    nadam = tf.keras.optimizers.Nadam(lr = 0.01)
    model.compile(optimizer = nadam, batch_size = 128, epochs = 30, validation_split = 0.1, callbacks = [reduceLR, checkpoint, earlystopping])

model.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_split = 0.1, callbacks = [reduceLR, checkpoint, earlystopping])