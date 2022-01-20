from cv2 import cv2

img = cv2.imread("cat.jpg")
print(img.shape)
img = img[1:-1, 1:-1]
print(img.shape)
