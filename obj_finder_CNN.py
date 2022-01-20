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

    def get_bbox(self, image):
        p = self.model.predict(np.expand_dims(image, axis=0))
        fmap = self.get_fmap(p[1][0])
        fmap = cv2.resize(fmap, self.input_shape, interpolation=cv2.INTER_NEAREST)
        # fmap = cv2.threshold(fmap, 200, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("fmap", fmap)
        # cv2.imshow("frame", frame)
        box = get_bbox((fmap == np.amax(fmap)).astype('uint8'))
        return int(p[0][0][0] * 100.0), box



if __name__ == '__main__':
    finder = ObjFinder(MODEL_PATH)

    # vs = JetsonVideoStream(outputResolution=finder.input_shape)
    # vs.start()
    # time.sleep(2.0)

    # print(model.layers[-2].get_weights())

    vs = cv2.VideoCapture(0)
    frame = vs.read()[1]

    while True:
        frame = vs.read()[1]
        frame = cv2.resize(frame, finder.input_shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''bboxes = finder.get_bboxes(fmap)
        for box in bboxes:
            frame = cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (255, 0, 0), 2)'''
        p, box = finder.get_bbox(frame)
        frame = cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (255, 0, 0), 2)
        cv2.imshow("roi", frame)
        if p > 50.0:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vs.release()
    # vs.stop()
    cv2.destroyAllWindows()
    time.sleep(1.0)
    sys.exit(0)
