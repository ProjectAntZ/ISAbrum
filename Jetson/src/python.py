import math
import struct
from time import sleep
import re

import cv2
import numpy as np
import serial
import tensorflow as tf
from imutils.video import JetsonVideoStream, VideoStream
from tqdm import trange

MODEL_PATH = "modelCNN"


def wait(s):
    for _ in trange(s - 1, bar_format='{n_fmt}/{total_fmt} seconds', initial=1, total=s):
        sleep(1)


def load_model(path):
    model = tf.keras.models.load_model(path)
    model.summary()

    last_conv = model.layers[-4].output
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

    def get_bbox(self, image, device='/gpu:0'):  # '/cpu:0'
        with tf.device(device):
            p = self.model.predict(np.expand_dims(image, axis=0))
        fmap = self.get_fmap(p[1][0])
        fmap = cv2.resize(fmap, self.input_shape, interpolation=cv2.INTER_NEAREST)
        # fmap = cv2.threshold(fmap, 200, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("fmap", fmap)
        box = get_bbox((fmap == np.amax(fmap)).astype('uint8'))
        return int(p[0][0][0] * 100.0), box


class Commander(serial.Serial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_text = ""
        self.flushInput()

    def send_ascii(self, text: str):
        if self.last_text != text or True:
            self.last_text = text
            ser.write(bytes(text, 'ascii'))


SENSORS_NUMBER = 3

ser = Commander(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
'''graph = RealtimeGraph(SENSORS_NUMBER)
graph.setScale(0, 200)
y = [0 for _ in range(SENSORS_NUMBER)]'''

finder = ObjFinder(MODEL_PATH)
vs = JetsonVideoStream(outputResolution=finder.input_shape)
vs.start()
wait(2)

HALF_WIDTH = finder.input_shape[0] // 2
TARGET_WIDTH = (HALF_WIDTH + 5, HALF_WIDTH + 20)
MIDDLE = (TARGET_WIDTH[0] + TARGET_WIDTH[1]) / 2.0
ADJUST_TIMES = (0.1, 1.0)

target_found = False

try:
    while True:
        if ser.in_waiting:
            print(ser.readall())
        '''msg = ser.readlines()
        for m in msg:
            m = m.decode('ascii')
            print(m)'''

        frame = vs.read()
        frame = cv2.resize(frame, finder.input_shape)
        # cv2.imshow("frame", frame)
        p, box = finder.get_bbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device='/cpu:0')
        frame = cv2.rectangle(frame, (box[2], box[0]), (box[3], box[1]), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (TARGET_WIDTH[0], 0), (TARGET_WIDTH[1], finder.input_shape[1]), (0, 255, 0), 2)
        x = (box[2] + box[3]) // 2
        # y = (box[0] + box[1]) // 2
        cv2.imshow("roi", frame)
        if p > 70 or target_found is True:
            print("Target found:", p)
            if x > TARGET_WIDTH[1] or x < TARGET_WIDTH[0]:
                ser.send_ascii('adjust\n')
                pos = x - MIDDLE  # 62.2 x 48.8 degrees

                sec = ADJUST_TIMES[1] * (MIDDLE / pos)
                if -ADJUST_TIMES[0] < sec < ADJUST_TIMES[1]:
                    sec = ADJUST_TIMES[0]

                ser.write(struct.pack('<f', sec))
                while True:
                    m = ser.readline().decode('ascii')
                    print(m)
                    if "finished" in m:
                        break
            else:
                ser.send_ascii('shoot\n')
                target_found = False

        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('s'):
                ser.send_ascii('shoot\n')
            elif key == ord('r'):
                ser.send_ascii('reload\n')
            elif key == ord('w'):
                ser.send_ascii('forward\n')
            elif key == ord('s'):
                ser.send_ascii('backward\n')
            elif key == ord('a'):
                ser.send_ascii('left\n')
            elif key == ord('d'):
                ser.send_ascii('right\n')
            elif key == ord('b'):
                ser.send_ascii('brake\n')
            elif key == ord('q'):
                break
except Exception as e:
    print(e)

vs.stop()
wait(2)
ser.close()
