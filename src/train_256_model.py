
"""
# Introduction

The U-Net model is a simple fully  convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. Mainly, it consists of two parts.

*   Contracting Path: we apply a series of conv layers and downsampling layers  (max-pooling) layers to reduce the spatial size
*   Expanding Path: we apply a series of upsampling layers to reconstruct the spatial size of the input.

The two parts are connected using a concatenation layers among different levels. This allows learning different features at different levels. At the end we have a simple conv 1x1 layer to reduce the number of channels to 1.
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import random
import cv2
from random import shuffle
import sys

# Import tensorflow/keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger # ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model


# Ensure tensorflow is actually using the GPU!!
tf.config.list_physical_devices('GPU')

"""
Note that we have two folders. The first one is `images` which contains the png images and `targets` which contains the masks.
"""

#constants
DATA_DIR = "patch_data_256"
PROBABALISTIC_LOSSES = ["binary_crossentropy", "binary_focal_crossentropy", "categorical_crossentropy", "categorical_focal_crossentropy", "sparse_categorical_crossentropy", "poisson", "kl_divergence", "ctc" ]

# Generators
def image_generator(files, batch_size = 32, sz = (256, 256)):

    while True:


        #extract a random batch
        batch = np.random.choice(files, size = batch_size)

        #variables for collecting batches of inputs and outputs
        batch_x = []
        batch_y = []

        for i, f in enumerate(batch):

            mask = Image.open(f'{DATA_DIR}/targets/{f}')
            mask = np.array(mask.resize(sz))

            batch_y.append(mask)

            #preprocess the raw images
            raw = Image.open(f'{DATA_DIR}/images/{f}')
            raw = raw.resize(sz) # @TODO -- figure out if I can remove this...
            raw = np.array(raw)

            #check the number of channels because some of the images are RGBA or GRAY
            if len(raw.shape) == 2:
              raw = np.stack((raw,)*3, axis=-1)

            else:
              raw = raw[:,:,0:3]

            batch_x.append(raw)

        #preprocess a batch of images and masks
        batch_x = np.array(batch_x)/255. # this makes them floats!!
        batch_y = np.array(batch_y)
        batch_y = np.expand_dims(batch_y,3)

        yield (batch_x, batch_y)


"""
SPLIT INTO BATCHES
"""

batch_size = 32

all_files = os.listdir(os.path.join('patch_data_256', 'images'))
shuffle(all_files)

split = int(0.95 * len(all_files))

#split into training and testing
train_files = all_files[0:split]
test_files  = all_files[split:]

train_generator = image_generator(train_files, batch_size = batch_size)
test_generator  = image_generator(test_files, batch_size = batch_size)


# @DEBUG
#print(test_generator)
#sys.exit(0)


""" @DEBUG
x, y= next(train_generator)

plt.axis('off')
img = x[0]
msk = y[0].squeeze()
msk = np.stack((msk,)*3, axis=-1)

plt.imshow( np.concatenate([img, msk, img*msk], axis = 1))
"""



def unet(sz = (256, 256, 3)):
    """
    ACTUAL MODEL!
    """
    x = Input(sz)
    inputs = x

    #down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = Conv2D(f, 3, activation='relu', padding='same') (x)
        layers.append(x)
        x = MaxPooling2D() (x)
        f = f*2
    ff2 = 64

    #bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1

    #upsampling
    for i in range(0, 5):
      ff2 = ff2//2
      f = f // 2
      x = Conv2D(f, 3, activation='relu', padding='same') (x)
      x = Conv2D(f, 3, activation='relu', padding='same') (x)
      x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
      x = Concatenate(axis=3)([x, layers[j]])
      j = j -1


    #classification
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    outputs = Conv2D(1, 1, activation='sigmoid') (x)

    #model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])

    return model


model = unet()

"""
TRAIN AND SAVE
"""
print("training!")
train_steps = len(train_files) // batch_size
test_steps = len(test_files) // batch_size
print(f"Train Steps: {train_steps}  Test Steps: {test_steps}")

# Callbacks
csv_logger = CSVLogger('models/256_training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
stop_early = EarlyStopping(patience=3)

model.fit(train_generator, epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator,
            validation_steps = test_steps, callbacks=[csv_logger, reduce_lr, stop_early], verbose=2)

model.get_weights()

#Save model
model.save(os.path.join("models", "256_unet.keras"))



"""
# References
#https://github.com/zaidalyafeai/Notebooks/blob/master/unet.ipynb
"""
