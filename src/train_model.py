# https://keras.io/examples/vision/oxford_pets_image_segmentation/

import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras import layers
import random

#debugging import
#from model_profiler import model_profiler



# losses used for classification
PROBABALISTIC_LOSSES = ["binary_crossentropy", "binary_focal_crossentropy", 
"poisson", "kl_divergence" ]

# "ctc" ]

""" for using extra labels...
"categorical_crossentropy", "categorical_focal_crossentropy", "sparse_categorical_crossentropy",
"""


#hinge loss?
# LOOK SPECIFICIALLY FOR IMAGE SEGMENTATION LOSS
# https://github.com/JunMa11/SegLossOdyssey


# make dice loss
# https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o


#gpu wizardry
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

"""

input_dir = "patch_data_256/images/"
target_dir = "patch_data_256/targets/"
img_size = (256, 256)
patch_size = img_size[0]
# only two after patch extractor
num_classes = 2 # 0-no damage, 1-minor, 2-major, 3-destroyed
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)



def get_datasets(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
):
    """Returns both TF Datasets."""

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        return input_img, target_img
        
    # dataset pipeline
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    splitpoint = int(len(dataset) * 0.8)
    train_ds = dataset.take(splitpoint)
    test_ds = dataset.skip(splitpoint)
    
    #train (cache THEN batch)
    train_ds = train_ds.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    #train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(len(train_ds))
    train_ds = train_ds.batch(batch_size)
    #train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
 
    #test (batch THEN cache)
    test_ds = test_ds.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    #test_ds = test_ds.cache()
    #test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


    return train_ds, test_ds


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Build model
model = get_model(img_size, num_classes)
model.summary()

#profile = model_profiler(model, batch_size)
#print(profile)

#exit()

train_ds, test_ds = get_datasets(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths
)



# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(
    optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy"
)

callbacks = [
    keras.callbacks.ModelCheckpoint(f"models/{patch_size}-model.keras", save_best_only=True),
    keras.callbacks.CSVLogger(f"models/f{patch_size}-training.log"),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.001),
    keras.callbacks.EarlyStopping(patience=3),
]

# Train the model, doing validation at the end of each epoch.
epochs = 30
model.fit(
    train_ds,
    epochs=epochs,
    validation_data=test_ds,
    callbacks=callbacks,
)

model.get_weights()

#Save model
model.save(os.path.join("models", f"{patch_size}_unet.keras"))


    