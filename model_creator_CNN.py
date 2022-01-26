import os

import tensorflow as tf
from tensorflow.keras import layers, activations

'''if tf.__version__ != "2.6.0":
    print("tf.__version__ != 2.6.0")
    exit(1)'''

from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_PATH = "/content/drive/MyDrive/Projects/autoHunter/modelCNN"
# MODEL_PATH = "model"

frameResolution = (320, 240)

frameResolution = (frameResolution[1], frameResolution[0])


def create_model():
    x = input = layers.Input(shape=frameResolution + (3,))
    x = layers.Rescaling(1. / 255)(x)

    size = [16, 32, 64, 128]
    for i, s in enumerate(size):
        for _ in range(1):
            x = layers.Conv2D(filters=s, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        if i != len(size)-1:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout2D(1. / 8)(x)
        x = layers.GaussianDropout(1. / 8)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(1. / 2)(x)
    x = layers.Dense(
        units=1,
        activation='sigmoid',
    )(x)

    output = x
    model = tf.keras.Model(inputs=input, outputs=output, name="Hunter")
    '''model = tf.keras.applications.mobilenet_v2(
        input_shape=frameResolution + (3,),
        alpha=1.0,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling='avg',
        classes=1,
        classifier_activation="sigmoid",
    )'''
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
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False,
        save_best_only=True,
    )

    model.fit(
        x=train,
        # steps_per_epoch=1000,
        validation_data=test,
        epochs=50,
        callbacks=[checkpoint]
    )
