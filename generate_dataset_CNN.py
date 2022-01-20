import os
from multiprocessing import Process

import numpy as np
from cv2 import cv2
from tqdm import tqdm

from dataset_generator_CNN import ObjectDetectionDatasetGenerator

frameResolution = (320, 240)

SOURCE_PATH = "D:/NN_DATA/x"
TARGET_PATH = "D:/NN_DATA/autoHunter"
TARGET_PATH_0 = os.path.join(TARGET_PATH, "0")
TARGET_PATH_1 = os.path.join(TARGET_PATH, "1")


def dataset_thread(dir_list, index):
    batch_size = 2

    gen = ObjectDetectionDatasetGenerator(
        obj_path="./objs",
        source_path=SOURCE_PATH,
        batch_size=batch_size,
        img_size=frameResolution,
        dir_list=dir_list
    )

    for i in tqdm(range(len(dir_list) // batch_size)):
        x, y = gen.__getitem__(i)
        for j, _ in enumerate(x):
            if y[j][0] == 0:
                f = os.path.join(TARGET_PATH_0, str(i) + '_' + str(index) + '.jpg')
            else:
                f = os.path.join(TARGET_PATH_1, str(i) + '_' + str(index) + '.jpg')

            cv2.imwrite(f, x[j].astype('uint8'))


if __name__ == '__main__':
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)
    if not os.path.exists(TARGET_PATH_0):
        os.mkdir(TARGET_PATH_0)
    if not os.path.exists(TARGET_PATH_1):
        os.mkdir(TARGET_PATH_1)

    n_p = 6
    source_list = os.listdir(SOURCE_PATH)
    # source_list = np.array(source_list).repeat(2).tolist()

    # slice_size = int(len(source_list)/(n_p*4))
    slice_size = 2000
    # dataset_thread(source_list[:slice_size], 1)

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
