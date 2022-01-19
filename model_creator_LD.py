import os
from multiprocessing import Process

import tensorflow as tf
from tensorflow.keras import layers

from dataset_generator_LD import LDDetectionDatasetGetter, get_splitted_getters

'''if tf.__version__ != "2.6.0":
    print("tf.__version__ != 2.6.0")
    exit(1)'''

MODEL_PATH = "/content/drive/MyDrive/Projects/autoHunter/modelLD"
# MODEL_PATH = "modelLD"

frameResolution = (320, 240)

frameResolution = (frameResolution[1], frameResolution[0])


def create_model():
    x = input = layers.Input(shape=frameResolution + (3,))
    # x = tf.keras.layers.LayerNormalization(axis=3)(x)

    size = [16, 32, 64, 128]
    for s in size:
        for _ in range(2):
            x = layers.Conv2D(s, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = tf.keras.layers.GaussianNoise(1. / 8)(x)
        if s != size[-1]:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(x)

    output = x
    model = tf.keras.Model(inputs=input, outputs=output, name="Hunter")
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='/content/drive/MyDrive/Projects/autoHunter/model.png', show_shapes=True)

    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics='mae',
    )
    print('LR: ', tf.keras.backend.get_value(model.optimizer.learning_rate))

    return model


def load_model(path):
    if os.path.exists(path) and str(input("Override? <Y/N>: ")).upper() != 'Y':
        model = tf.keras.models.load_model(path)
        model.summary()
        print('Old LR: ', tf.keras.backend.get_value(model.optimizer.learning_rate))
        tf.keras.backend.set_value(model.optimizer.learning_rate, float(input('LR: ')))
    else:
        print("Creating new model...")
        model = create_model()
    return model


if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    DATA_PATH = "/content/data/autoHunter"
    # DATA_PATH = "D:/NN_DATA/autoHunter"

    train, test = get_splitted_getters(
        source_path=DATA_PATH,
        input_shape=frameResolution + (3,),
        output_shape=tuple(model.output.shape[1:3]),
        split=0.8,
        batch_size=16,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_mae',
        mode='max',
        save_weights_only=False,
        save_best_only=True,
    )

    model.fit(
        x=train,
        validation_data=test,
        epochs=25,
        callbacks=[checkpoint]
    )
