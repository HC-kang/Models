import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPool2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Input

(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'Cifar10',
    split = ['train', 'test'],
    batch_size = -1,
    as_supervised = True,
))
label_train = tf.one_hot(label_train, depth = 10)
label_test = tf.one_hot(label_test, depth = 10)

def Conv_layer1(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    shortcut = x
    
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    shortcut = x
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def Conv_layer2(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    shortcut = x
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', strides = 2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x1 = Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', strides = 2)(shortcut)
    x1 = BatchNormalization()(x1)
    x = Add()([x, x1])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def Conv_layer3(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', strides = 2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x1 = Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', strides = 2)(shortcut)
    x1 = BatchNormalization()(x1)
    x = Add()([x, x1])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

inputs = Input(shape = (32, 32, 3), dtype = 'float32')
x = Conv_layer1(inputs)
x = Conv_layer2(x)
x = Conv_layer3(x)
x = GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x)
resnet20 = tf.keras.models.Model(inputs, outputs)
resnet20.summary()

nadam = tf.keras.optimizers.Nadam(lr = 0.01)
resnet20.compile(optimizer = nadam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
resnet20.fit(img_train, label_train, batch_size = 128, epochs = 10, validation_split = 0.1)