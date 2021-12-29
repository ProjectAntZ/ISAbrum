from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

from image_transformer import add_RGBA2RGB

cam = cv2.VideoCapture(0)

template = cv2.imread('objs/obj.png', flags=cv2.IMREAD_UNCHANGED)
noise = np.random.normal(0,1,template.shape[:2] + (3,)).astype('uint8')

template, _ = add_RGBA2RGB(rgb=noise, rgba=template)
template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

w, h = template.shape[::-1]

while True:
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        method = eval(meth)
        # Apply template Matching

        res = cv2.matchTemplate(frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(frame, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
