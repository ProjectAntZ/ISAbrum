import numpy as np
from cv2 import cv2

cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ret, labels = cv2.connectedComponents(frame)

    frame = frame / 255.0

    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    r -= b * 0.5 + g * 0.5
    print(np.max(r))

    if np.max(r) > 0.5:
        r = (r > np.max(r) * 0.8) * 1.0
        print(np.argmax(r))

    cv2.imshow("a", r)
    cv2.waitKey(1)
