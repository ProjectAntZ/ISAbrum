import sys
import time

import numpy as np
import tensorflow as tf
from cv2 import cv2

# from imutils.video import JetsonVideoStream, VideoStream

MODEL_PATH = "modelCNN"


def load_model(path):
    model = tf.keras.models.load_model(path)

    last_conv = model.layers[-8].output
    gp = model.layers[-5].output
    print(last_conv)
    print(gp)

    '''x = last_conv.output
    x = tf.math.reduce_sum(x, axis=3)
    x = scale_min_max(x, 0.0, 255.0)
    x = tf.cast(x, dtype=tf.uint8)'''
    # x = tf.math.greater(x, 120)

    model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.output,
            last_conv,
            gp
        ]
    )

    return model


frameResolution = (320, 240)

if __name__ == '__main__':
    # vs = JetsonVideoStream(outputResolution=frameResolution)
    # vs.start()
    # time.sleep(2.0)
    model = load_model(MODEL_PATH)
    # print(model.layers[-2].get_weights())

    cam = cv2.VideoCapture(0)
    _, frame = cam.read()

    while True:
        # frame = vs.read()
        _, frame = cam.read()
        frame = cv2.resize(frame, frameResolution)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        p = model.predict(np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0))
        cv2.imshow("video", frame)

        roi = p[1][0]
        roi = np.multiply(roi, p[2][0])
        roi = np.sum(roi, axis=2)
        roi = roi/np.amax(roi)*255.0
        roi = cv2.resize(roi.astype('uint8'), (frame.shape[1], frame.shape[0]))
        cv2.imshow("roi", roi)
        print(p[0][0]*100)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    # vs.stop()
    time.sleep(1.0)
    sys.exit(0)
