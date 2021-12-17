import os

import tensorflow as tf

'''if tf.__version__ != "2.6.0":
    print("tf.__version__ != 2.6.0")
    exit(1)'''

from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_PATH = "/content/drive/MyDrive/Projects/autoHunter/modelCNN"
# MODEL_PATH = "model"

frameResolution = (320, 240)

frameResolution = (frameResolution[1], frameResolution[0])


def create_model():
    x = input = tf.keras.layers.Input(shape=frameResolution + (3,))
    x = tf.keras.layers.Rescaling(1. / 255)(x)
    # x = tf.keras.layers.LayerNormalization(axis=3)(x)

    size = [3, 2]
    for i, s in enumerate(size):
        for _ in range(1):
            x = tf.keras.layers.DepthwiseConv2D(depth_multiplier=s, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.keras.layers.GaussianDropout(1. / 8)(x)
        if i != len(size)-1:
            # x = tf.keras.layers.SpatialDropout2D(1. / 8)(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # x = tf.keras.layers.GaussianDropout(1. / 8)(x)

    '''x = input = tf.keras.layers.Input(shape=frameResolution + (3,))
    x = tf.keras.layers.Rescaling(1. / 255)(x)

    size = [32, 64, 128, 256]
    for i, s in enumerate(size):
        for _ in range(1):
            x = tf.keras.layers.Conv2D(filters=s, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.GaussianDropout(1. / 8)(x)
            if i != len(size)-1:
                x = tf.keras.layers.SpatialDropout2D(1. / 4)(x)
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)'''

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(1. / 2)(x)
    x = tf.keras.layers.Dense(
        units=32,
        activation='relu',
    )(x)
    x = tf.keras.layers.Dropout(1. / 4)(x)
    x = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
    )(x)

    output = x
    model = tf.keras.Model(inputs=input, outputs=output, name="Hunter")
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='/content/drive/MyDrive/Projects/autoHunter/model.png', show_shapes=True)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics='accuracy',
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
    train = image_dataset_from_directory(
        directory=TARGET_PATH,
        labels='inferred',
        label_mode='binary',
        batch_size=8,
        image_size=frameResolution,
        shuffle=True,
        subset="training",
        validation_split=0.1,
        seed=1234,
        color_mode="rgb",
    )
    test = image_dataset_from_directory(
        directory=TARGET_PATH,
        labels='inferred',
        label_mode='binary',
        batch_size=8,
        image_size=frameResolution,
        shuffle=True,
        subset="validation",
        validation_split=0.1,
        seed=1234,
        color_mode="rgb",
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            save_weights_only=False,
            save_best_only=False
    )

    model.fit(
        x=train,
        # steps_per_epoch=1000,
        validation_data=test,
        epochs=20,
        callbacks=[checkpoint]
    )
