import sys
import time

import numpy as np
import tensorflow as tf
from cv2 import cv2

# from imutils.video import JetsonVideoStream, VideoStream

MODEL_PATH = "modelCNN"


def load_model(path):
    model = tf.keras.models.load_model(path)
    model.summary()

    last_conv = model.layers[-5].output
    gp = model.layers[-3].output
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


def get_bbox(image):
    a = np.where(image != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


class ObjFinder:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.input_shape = tuple(reversed(self.model.input.shape[1:3]))

    def get_fmap(self, maps):
        for i in range(maps.shape[2]):
            maps[:, :, i] -= np.amin(maps[:, :, i])
            if np.amax(maps[:, :, i]) != 0:
                maps[:, :, i] /= np.amax(maps[:, :, i])

        # roi = np.multiply(roi, p[2][0])
        maps = np.sum(maps, axis=2)
        maps = maps / np.amax(maps) * 255.0
        return maps.astype('uint8')

    def get_bboxes(self, fmap):
        boxes = []
        num_labels, labels = cv2.connectedComponents(fmap)
        for l in range(num_labels):
            bbox = get_bbox((labels == l).astype('uint8'))
            boxes.append(bbox)

        return boxes


if __name__ == '__main__':
    # vs = JetsonVideoStream(outputResolution=frameResolution)
    # vs.start()
    # time.sleep(2.0)
    finder = ObjFinder(MODEL_PATH)
    # print(model.layers[-2].get_weights())

    cam = cv2.VideoCapture(0)
    _, frame = cam.read()

    while True:
        # frame = vs.read()
        _, frame = cam.read()
        frame = cv2.resize(frame, finder.input_shape)

        p = finder.model.predict(np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0))
        print(int(p[0][0][0] * 100.0))
        fmap = finder.get_fmap(p[1][0])
        fmap = cv2.resize(fmap, finder.input_shape, interpolation=cv2.INTER_NEAREST)
        fmap = cv2.threshold(fmap, 100, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("fmap", fmap)
        cv2.imshow("frame", frame)
        bboxes = finder.get_bboxes(fmap)
        for box in bboxes:
            frame = cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (255, 0, 0), 2)

        cv2.imshow("roi", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    # vs.stop()
    time.sleep(1.0)
    sys.exit(0)
