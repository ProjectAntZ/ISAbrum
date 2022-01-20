import time

import cv2
import numpy as np
import serial
import tensorflow as tf
from imutils.video import JetsonVideoStream, VideoStream

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
        #cv2.imshow("fmap", fmap)
        box = get_bbox((fmap == np.amax(fmap)).astype('uint8'))
        return int(p[0][0][0] * 100.0), box


SENSORS_NUMBER = 3

ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
# graph = RealtimeGraph(SENSORS_NUMBER)
# graph.setScale(0, 200)
# y = [0 for _ in range(SENSORS_NUMBER)]

finder = ObjFinder(MODEL_PATH)
vs = JetsonVideoStream(outputResolution=finder.input_shape)
vs.start()
time.sleep(2.0)

TARGET_WIDTH = (finder.input_shape[0] // 2 - 10, finder.input_shape[0] // 2 + 20)

try:
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, finder.input_shape)
        #cv2.imshow("frame", frame)
        p, box = finder.get_bbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (TARGET_WIDTH[0], 0), (TARGET_WIDTH[1], finder.input_shape[1]), (0, 255, 0), 2)
        x = (box[2] + box[3]) // 2
        # y = (box[0] + box[1]) // 2
        cv2.imshow("roi", frame)
        if p > 50.0:
            if x > TARGET_WIDTH[1]:
                ser.write(bytes('adjustRight\n', 'ascii'))
            elif x < TARGET_WIDTH[0]:
                ser.write(bytes('adjustLeft\n', 'ascii'))
            else:
                ser.write(bytes('shoot\n', 'ascii'))
        else:
            ser.write(bytes('targetEliminated\n', 'ascii'))

        msg = ser.readlines()
        for m in msg:
            m = m.decode('ascii')
            '''if 'Sensor: ' in msg:
                s = int(re.search('Sensor: (\d+)', msg).group(1))
                d = int(re.search('Distance: (\d+)', msg).group(1))
                y[s] = d
                graph.show(y)'''
            print(m)

        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('s'):
                ser.write(bytes('shoot\n', 'ascii'))
            elif key == ord('r'):
                ser.write(bytes('reload\n', 'ascii'))
            elif key == ord('q'):
                break
except KeyboardInterrupt:
    print('Interrupted')

vs.stop()
time.sleep(1.0)
ser.close()
