import math
import os
import random

import numpy as np
from cv2 import cv2

import image_transformer as it
import tensorflow as tf


def _transform_object(obj, rotation, scaling):
    obj = it.transform(obj, rotation=rotation, scaling=scaling)
    return obj.astype('uint8')


def _add_object_to_image(obj, img, scale=(0.0, 1.0)):
    while True:
        scale_factor = random.uniform(scale[0], scale[1])

        obj_height = int(img.shape[0] * scale_factor)
        obj_width = int(obj_height * (obj.shape[1] / obj.shape[0]))
        obj = cv2.resize(obj, (obj_width, obj_height))

        if img.shape[1] - obj.shape[1] > 0 and img.shape[0] - obj.shape[0] > 0:
            x_offset = random.randint(0, img.shape[1] - obj.shape[1])
            y_offset = random.randint(0, img.shape[0] - obj.shape[0])

            img, _ = it.add_RGBA2RGB(rgb=img, rgba=obj, x_offset=x_offset, y_offset=y_offset)
            break
    return img.astype('uint8')


def _add_noises(img, contrast, brightness, balance):
    for i in random.sample(range(5), 5):
        if i == 0:
            img = cv2.blur(img, (random.randint(1, 3), random.randint(1, 3)))
        elif i == 1:
            img = it.add_gaussian_noise(img, random.uniform(0.01, 0.1))
        elif i == 2:
            img = it.random_contrast(image=img, a=contrast[0], b=contrast[1])
        elif i == 3:
            img = it.random_brightness(image=img, a=brightness[0], b=brightness[1])
        elif i == 4:
            img = it.random_color_balance(image=img, a=balance[0], b=balance[1])

    return img.astype('uint8')


def _resize_prop(img, size):
    pass


class ObjectDetectionDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, obj_path, source_path, img_size, batch_size=2, n_images=None, log_path=None, dir_list=None):
        obj_img_dir = [p for p in os.listdir(obj_path) if p.split('.')[-1] == "png"]

        self.obj_img = [cv2.imread(filename=os.path.join(obj_path, img_name), flags=cv2.IMREAD_UNCHANGED) for img_name in obj_img_dir]

        new_height = img_size[0]
        for i, img in enumerate(self.obj_img):
            new_width = int(img_size[1] * img.shape[1] / img.shape[0])
            self.obj_img[i] = cv2.resize(img, (new_width, new_height))

        self.source_path = source_path

        if dir_list is not None:
            self.source_images_names = dir_list
        else:
            self.source_images_names = os.listdir(self.source_path)

        self.n_images = n_images
        if n_images is None or self.n_images > len(self.source_images_names):
            self.n_images = len(self.source_images_names)
        self.source_images_names = self.source_images_names[:self.n_images]

        self.on_epoch_end()

        self.batch_size = batch_size

        self.img_size = img_size
        self.labels_dev = 2
        self.contrast = (0.8, 1.2)
        self.brightness = (0.8, 1.2)
        self.scale = (0.3, 0.7)
        self.balance = (0.8, 1.2)

        self.log_path = log_path

    def __len__(self):
        return int(math.floor(len(self.source_images_names) / self.batch_size))

    def __getitem__(self, index):
        x = []
        y = []

        for count, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size, 1)):
            img = cv2.imread(os.path.join(self.source_path, self.source_images_names[i]))

            img = cv2.resize(img, self.img_size)

            img, label = self._next(img, i)
            x.append(img)
            y.append(label)

        return x, y

    def on_epoch_end(self):
        random.shuffle(self.source_images_names)

    def _next(self, random_image, index):
        label = 0

        if index % self.labels_dev == 1:
            label = 1
            obj_img_copy = self.obj_img[random.randint(0, len(self.obj_img) - 1)].copy()

            rotation = (random.randint(-10, 10), random.randint(-10, 10), random.randint(-20, 20))
            scaling = (random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))

            obj_img_copy = _transform_object(obj_img_copy, rotation, scaling)

            kernel = np.ones((3, 3), np.uint8)
            obj_img_copy[:, :, 3] = cv2.erode(obj_img_copy[:, :, 3], kernel, iterations=1)

            random_image = _add_object_to_image(obj=obj_img_copy, img=random_image, scale=self.scale)

            if random.randint(0, 1) == 1:
                random_image = cv2.flip(random_image, 1)

        '''elif index % 5 == 1:
            obj_img_copy = self.obj_img.copy()

            if random.randint(0, 1) == 0:
                obj_img_copy = cv2.rotate(obj_img_copy, cv2.ROTATE_90_CLOCKWISE)
            else:
                obj_img_copy = cv2.rotate(obj_img_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rotation = (random.randint(-5, 5), random.randint(-5, 5), random.randint(-15, 15))
            scaling = (random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))

            obj_img_copy = _transform_object(obj_img_copy, rotation, scaling)
            random_image = _add_object_to_image(obj=obj_img_copy, img=random_image, scale=self.scale)'''

        random_image = _add_noises(
            img=random_image,
            brightness=self.brightness,
            contrast=self.contrast,
            balance=self.balance
        )
        return random_image, [label]
