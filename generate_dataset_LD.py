import os
from multiprocessing import Process

import numpy as np
from cv2 import cv2
from tqdm import tqdm

from dataset_generator_LD import LDDetectionDatasetGenerator

frameResolution = (320, 240)

SOURCE_PATH = "D:/NN_DATA/x"
TARGET_PATH = "D:/NN_DATA/autoHunter"
TARGET_PATH_X = os.path.join(TARGET_PATH, "x")
TARGET_PATH_Y = os.path.join(TARGET_PATH, "y")


def dataset_thread(dir_list, index):
    gen = LDDetectionDatasetGenerator(
        obj_path="objs",
        source_path=SOURCE_PATH,
        img_size=frameResolution,
        dir_list=dir_list
    )

    for i in tqdm(range(len(dir_list))):
        x, y = gen.__getitem__(i)
        x_path = os.path.join(TARGET_PATH_X, str(index) + '_' + str(i) + '.jpg')
        cv2.imwrite(x_path, x.astype('uint8'))

        y_path = os.path.join(TARGET_PATH_Y, str(index) + '_' + str(i) + '.jpg')
        cv2.imwrite(y_path, y.astype('uint8'))


if __name__ == '__main__':
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)
    if not os.path.exists(TARGET_PATH_X):
        os.mkdir(TARGET_PATH_X)
    if not os.path.exists(TARGET_PATH_Y):
        os.mkdir(TARGET_PATH_Y)

    n_p = 6
    source_list = os.listdir(SOURCE_PATH)
    # source_list = np.array(source_list).repeat(2).tolist()

    # slice_size = int(len(source_list) / (n_p ** 2))
    slice_size = 1000

    processes = []
    for n in range(n_p):
        processes.append(
            Process(
                target=dataset_thread,
                args=(source_list[:slice_size], n)
            )
        )
        source_list = source_list[slice_size:]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
