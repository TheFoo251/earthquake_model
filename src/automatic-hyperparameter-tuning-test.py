# try out automatic tuning

import tensorflow as tf
import keras_tuner as kt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

(x_train.shape, y_train.shape), (x_test.shape, y_test.shape)

#preprocess
x_train = x_train / 255.0
x_test = x_test / 255.0

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    # what is the best activation function?
    hp_activation = hp.Choice("activation", values=["relu", "tanh"])
    hp_layer_1 = hp.Int("layer_1", min_value=1, max_value=1000, step=100)
    hp_layer_2 = hp.Int("layer_2", min_value=1, max_value=1000, step=100)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
    model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
        
    return model
    

tuner = kt.Hyperband(model_builder,
                    objective="val_accuracy",
                    max_epochs=10,
                    factor=3,
                    directory="dir",
                    project_name='x')
                    
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

