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


def _add_noises(img, contrast, brightness):
    for i in random.sample(range(4), 4):
        if i == 0:
            img = cv2.blur(img, (random.randint(1, 2), random.randint(1, 2)))
        elif i == 1:
            img = it.add_gaussian_noise(img, random.uniform(0.01, 0.1))
        elif i == 2:
            img = it.random_contrast(image=img, a=contrast[0], b=contrast[1])
        elif i == 3:
            img = it.random_brightness(image=img, a=brightness[0], b=brightness[1])

    return img.astype('uint8')


def _resize_prop(img, size):
    pass


class ObjectDetectionDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, obj_path, source_path, img_size, batch_size=2, n_images=None, log_path=None, dir_list=None):
        self.obj_img = cv2.imread(filename=obj_path, flags=cv2.IMREAD_UNCHANGED)
        new_heigth = img_size[0]
        new_width = int(img_size[1] * self.obj_img.shape[1] / self.obj_img.shape[0])
        self.obj_img = cv2.resize(self.obj_img, (new_width, new_heigth))

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
        self.contrast = (0.6, 1.2)
        self.brightness = (0.7, 1.2)
        self.scale = (0.4, 0.7)

        self.log_path = log_path

    def __len__(self):
        return int(math.floor(len(self.source_images_names) / self.batch_size))

    def __getitem__(self, index):
        x = []
        y = []

        for count, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size, 1)):
            img = cv2.imread(os.path.join(self.source_path, self.source_images_names[i]))
            img = cv2.resize(img, self.img_size)

            target_x, target_y = self._next(img, i)

            y.append([target_y])
            x.append(target_x)

        offset = index * self.batch_size
        if self.log_path is not None:
            for i in range(self.labels_dev):
                path = os.path.join(self.log_path, str(y[i]) + "_" + self.source_images_names[offset + i])
                cv2.imwrite(path, x[i])

        return x, y

    def on_epoch_end(self):
        random.shuffle(self.source_images_names)

    def _next(self, random_image, index):
        label = 0

        if index % self.labels_dev == 1:
            label = 1
            obj_img_copy = self.obj_img.copy()

            rotation = (random.randint(-5, 5), random.randint(-5, 5), random.randint(-15, 15))
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

        random_image = _add_noises(img=random_image, brightness=self.brightness, contrast=self.contrast)
        return random_image, label
