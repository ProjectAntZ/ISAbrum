import time

import cv2
from imutils.video import JetsonVideoStream, VideoStream

vs = JetsonVideoStream()
vs.start()
time.sleep(2.0)

while True:
    cv2.imshow("", vs.read())
    if cv2.waitKey(1) == ord('q'):
        break

vs.stop()
time.sleep(1.0)
