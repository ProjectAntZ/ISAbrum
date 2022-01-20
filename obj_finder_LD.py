import sys
import time

import numpy as np
import tensorflow as tf
from cv2 import cv2
# from imutils.video import JetsonVideoStream, VideoStream

MODEL_PATH = "modelLD"


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def get_bbox(image):
    a = np.where(image != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


frameResolution = (320, 240)

if __name__ == '__main__':
    #vs = JetsonVideoStream(outputResolution=frameResolution)
    #vs.start()
    #time.sleep(2.0)
    model = load_model(MODEL_PATH)

    cam = cv2.VideoCapture(0)
    _, frame = cam.read()

    while True:
        #frame = vs.read()
        _, frame = cam.read()
        frame = cv2.resize(frame, frameResolution)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        roi = model.predict(np.expand_dims(frame, axis=0))[0]
        roi = (((roi - np.amin(roi)) / np.amax(roi)) * 255.0).astype('uint8')
        roi = cv2.resize(roi, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        #bbox = get_bbox(roi)

        #frame = cv2.rectangle(frame, bbox[0:2], bbox[2:4], (255, 0, 0), 2)
        cv2.imshow("video", frame)
        cv2.imshow("roi", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    # vs.stop()
    time.sleep(1.0)
    sys.exit(0)
