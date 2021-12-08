from cv2 import cv2
import image_transformer as it
import random

contrast = (0.6, 1.2)
brightness = (0.8, 1.2)

img=cv2.imread("cat.jpg")
cv2.imshow("img", img)
cv2.imshow("blur", cv2.blur(img, (3, 3)))
cv2.imshow("add_gaussian_noise", it.add_gaussian_noise(img, 0.05))
cv2.imshow("add_noise", it.add_noise(img))

'''img = it.random_contrast(image=img, a=contrast[0], b=contrast[1])
cv2.imshow("random_contrast", img)
img = it.random_brightness(image=img, a=brightness[0], b=brightness[1])
cv2.imshow("random_brightness", img)'''

cv2.waitKey()
