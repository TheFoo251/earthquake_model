import numpy as np
import tensorflow as tf
from tensorflow import keras

data_dir = os.path.join(data)

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    directory,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode="rgba",
    batch_size=32,
    image_size=(1024, 1024),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="both"
)


