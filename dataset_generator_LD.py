import math
import os
import random
from time import sleep

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

            img, land = it.add_RGBA2RGB(rgb=img, rgba=obj, x_offset=x_offset, y_offset=y_offset)
            break

    return img.astype('uint8'), land.astype('uint8')


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


class LDDetectionDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, obj_path, source_path,  img_size, batch_size=2, n_images=None, dir_list=None):
        obj_img_dir = [p for p in os.listdir(obj_path) if p.split('.')[-1] == "png"]

        self.obj_img = [cv2.imread(filename=os.path.join(obj_path, img_name), flags=cv2.IMREAD_UNCHANGED) for img_name
                        in obj_img_dir]

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
        self.contrast = (0.7, 1.2)
        self.brightness = (0.7, 1.2)
        self.scale = (0.4, 0.8)

    def __len__(self):
        return int(math.floor(len(self.source_images_names) / self.batch_size))

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.source_path, self.source_images_names[index]))

        img = cv2.resize(img, self.img_size)

        target_x, target_y = self._next(img, index)
        return target_x, target_y

    def on_epoch_end(self):
        random.shuffle(self.source_images_names)

    def _next(self, random_image, index):
        obj_img_copy = self.obj_img[random.randint(0, len(self.obj_img) - 1)].copy()

        rotation = (random.randint(-10, 10), random.randint(-10, 10), random.randint(-20, 20))
        scaling = (random.uniform(0.9, 1.2), random.uniform(0.9, 1.2), random.uniform(0.9, 1.2))

        obj_img_copy = _transform_object(obj_img_copy, rotation, scaling)
        kernel = np.ones((3, 3), np.uint8)
        obj_img_copy[:, :, 3] = cv2.erode(obj_img_copy[:, :, 3], kernel, iterations=1)

        random_image, ld = _add_object_to_image(obj=obj_img_copy, img=random_image, scale=self.scale)
        random_image = _add_noises(img=random_image, brightness=self.brightness, contrast=self.contrast)
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

        return random_image, ld


def get_splitted_getters(source_path, split, input_shape, output_shape, batch_size):
    train = LDDetectionDatasetGetter(source_path, input_shape, output_shape, split=split, batch_size=batch_size)
    test = LDDetectionDatasetGetter(source_path, input_shape, output_shape, split=1-split, batch_size=batch_size)
    return train, test


class LDDetectionDatasetGetter(tf.keras.utils.Sequence):
    def __init__(self, source_path, input_shape, output_shape, batch_size, n_images=None, dir_list=None, split=None):
        self.source_path = source_path
        self.source_path_x = os.path.join(self.source_path, "x")
        self.source_path_y = os.path.join(self.source_path, "y")

        self.batch_size = batch_size

        if dir_list is not None:
            self.source_images_names = dir_list
        else:
            self.source_images_names = os.listdir(self.source_path_x)

        self.n_images = n_images
        if n_images is None or self.n_images > len(self.source_images_names):
            self.n_images = len(self.source_images_names)
        self.source_images_names = self.source_images_names[:int(self.n_images * split)]

        self.input_shape = input_shape
        self.output_shape = output_shape

    def __len__(self):
        return len(self.source_images_names) // self.batch_size

    def __getitem__(self, index):
        x = []
        y = []

        for count, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size, 1)):
            x_img = cv2.imread(os.path.join(self.source_path_x, self.source_images_names[i]))
            y_img = cv2.imread(os.path.join(self.source_path_y, self.source_images_names[i]), cv2.IMREAD_GRAYSCALE)

            x_img = cv2.resize(x_img, tuple(reversed(self.input_shape[:2])), interpolation=cv2.INTER_LINEAR)
            y_img = cv2.resize(y_img, tuple(reversed(self.output_shape[:2])), interpolation=cv2.INTER_NEAREST)

            x.append(x_img)
            y.append(y_img)

        x = np.array(x) / 255.0
        y = np.expand_dims(y, axis=-1) / 255.0
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.source_images_names)
