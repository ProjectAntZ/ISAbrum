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

    last_conv = model.layers[-7].output
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
        fmap = cv2.threshold(fmap, 100, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels = cv2.connectedComponents(fmap)
        for l in range(num_labels):
            contours = cv2.findContours((labels == l).astype('uint8') * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            boxes.append(cv2.boundingRect(contours))

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
        fmap = cv2.resize(fmap, finder.input_shape)
        bboxes = finder.get_bboxes(fmap)
        for box in bboxes:
            frame = cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)

        cv2.imshow("roi", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    # vs.stop()
    time.sleep(1.0)
    sys.exit(0)
