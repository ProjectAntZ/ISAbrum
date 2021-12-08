import os
from multiprocessing import Process

import tensorflow as tf

from dataset_generator_LD import LDDetectionDatasetGetter, get_splitted_getters

'''if tf.__version__ != "2.6.0":
    print("tf.__version__ != 2.6.0")
    exit(1)'''

MODEL_PATH = "/content/drive/MyDrive/Projects/autoHunter/modelLD"
# MODEL_PATH = "modelLD"

frameResolution = (320, 240)

frameResolution = (frameResolution[1], frameResolution[0])


def create_model():
    x = input = tf.keras.layers.Input(shape=frameResolution + (3,))
    x = tf.keras.layers.Rescaling(1. / 255)(x)
    # x = tf.keras.layers.LayerNormalization(axis=3)(x)

    poll_layers = []
    size = [32, 64, 128, 256]
    for s in size:
        for _ in range(2):
            x = tf.keras.layers.Conv2D(s, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.GaussianNoise(1. / 16)(x)
        if s != size[-1]:
            poll_layers.append(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for s in reversed(size):
        if s != size[-1]:
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            x = tf.keras.layers.Concatenate()([poll_layers[-1], x])
            poll_layers.pop()

        for _ in range(2):
            x = tf.keras.layers.Conv2D(s, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.GaussianNoise(1. / 16)(x)

    x = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    output = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    model = tf.keras.Model(inputs=input, outputs=output, name="Hunter")
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='/content/drive/MyDrive/Projects/autoHunter/model.png', show_shapes=True)

    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
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
    TARGET_PATH = "/content/data/autoHunter"
    # TARGET_PATH = "D:/NN_DATA/autoHunter"
    '''gen = LDDetectionDatasetGetter(
        source_path=TARGET_PATH,
        input_shape=frameResolution + (3,),
        output_shape=tuple(model.output.shape[1:3])
    )'''

    train, test = get_splitted_getters(
        source_path=TARGET_PATH,
        input_shape=frameResolution + (3,),
        output_shape=tuple(model.output.shape[1:3]),
        split=0.8,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            save_weights_only=False,
    )

    model.fit(
        x=train,
        validation_data=test,
        epochs=4,
        callbacks=[checkpoint]
    )
