import random

from cv2 import cv2
import numpy as np


def random_color_balance(image, a, b):
    balance = random.uniform(a, b)
    image2 = np.zeros(image.shape)
    image2[:,:,0] = ((1 + 2*balance)*image[:,:,0] + (1 - balance)*image[:,:,1] + (1 - balance)*image[:,:,2])/3
    image2[:,:,1] = ((1 + 2*balance)*image[:,:,1] + (1 - balance)*image[:,:,0] + (1 - balance)*image[:,:,2])/3
    image2[:,:,2] = ((1 + 2*balance)*image[:,:,2] + (1 - balance)*image[:,:,0] + (1 - balance)*image[:,:,1])/3
    image2 = image2.astype('uint8')
    return image2


def add_gaussian_noise(image, f):
    image = image + np.random.normal(0, 255 * f, image.shape)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image.astype('uint8')


def add_noise(image):
    return np.clip(image + (np.std(image) * np.random.random(size=image.shape[:3])), 0, 255)


def random_contrast(image, a, b):
    return np.clip(image * random.uniform(a, b), 0, 255)


def random_brightness(image, a, b):
    return np.clip(image + (255 * (random.uniform(a, b) - 1)), 0, 255)


def resize_prop(image, width, height, color=None):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    # dest = np.clip(np.random.random(size=(height, width, image.shape[2]))*255, 0, 255)
    if color:
        dest = np.zeros(shape=(height, width, image.shape[2]), dtype='uint8')
        dest[:, :] = color
    else:
        dest = np.zeros(shape=(height, width, image.shape[2]), dtype='uint8')

    img_width = image.shape[1]
    img_height = image.shape[0]

    if (width / height) >= (img_width / img_height):
        image = cv2.resize(image, (int(height * (img_width / img_height)), height))
    else:
        image = cv2.resize(image, (width, int(width * (img_height / img_width))))

    img_width = image.shape[1]
    img_height = image.shape[0]

    x1 = int((width - img_width) / 2)
    x2 = int((width - img_width) / 2 + img_width)

    y1 = int((height - img_height) / 2)
    y2 = int((height - img_height) / 2 + img_height)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    dest[y1:y2, x1:x2, :] = image

    return dest, {"x": x1, "y": y1}, {"x": x2, "y": y2}


def draw_lines(image, points, color=(0, 255, 0)):
    for i in range(0, len(points) - 2, 2):
        image = cv2.line(image,
                         (int(points[i]), int(points[i + 1])), (int(points[i + 2]), int(points[i + 3])),
                         color)

    image = cv2.line(image,
                     (int(points[len(points) - 2]), int(points[len(points) - 1])), (int(points[0]), int(points[1])),
                     color)

    return image


def get_alpha_corners(alpha):
    corners = cv2.goodFeaturesToTrack(alpha, maxCorners=4, qualityLevel=0.1, minDistance=1)[:, 0]
    if corners.shape[0] * corners.shape[1] != 8:
        return None

    corners = corners[corners[:, 0].argsort()]
    if corners[0, 1] < corners[1, 1]:
        corners[[0, 1]] = corners[[1, 0]]

    if corners[2, 1] < corners[3, 1]:
        corners[[2, 3]] = corners[[3, 2]]

    r = np.array([])
    for c in corners:
        r = np.append(r, c[0])
        r = np.append(r, c[1])

    return r


def offset_corners(corners, x_offset=0, y_offset=0):
    moved_corners = np.array([])
    for i, c in enumerate(corners):
        if i % 2 == 0:
            moved_corners = np.append(moved_corners, c + x_offset)
        else:
            moved_corners = np.append(moved_corners, c + y_offset)

    return moved_corners


def add_alpha(image):
    if image.shape[2] < 4:
        b_channel, g_channel, r_channel = cv2.split(image)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.

        alpha_channel[0, :] = 0
        alpha_channel[:, 0] = 0
        alpha_channel[b_channel.shape[0] - 1, :] = 0
        alpha_channel[:, b_channel.shape[1] - 1] = 0

        image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return image


def add_RGBA2RGB(rgb, rgba, x_offset=0, y_offset=0):
    h, w = rgb.shape[:2]
    ah, aw = rgba.shape[:2]

    for y, yp in enumerate(rgba):
        for x, xp in enumerate(yp):
            if xp[3] != 0:
                rgb[y_offset + y, x_offset + x] = xp[:3]

    a = np.zeros((h, w), np.uint8)
    a[y_offset:y_offset + ah, x_offset:x_offset + aw] = rgba[:, :, 3]

    return rgb, a


def const_resize(image, size):
    if image.shape[2] == 4:
        h, w = image.shape[:2]
        blank_h, blank_w = int(size), int(size)
        blank = np.zeros([blank_h, blank_w, 4], dtype=np.uint8)
        blank[:, :, 3] = 0
        blank[int((blank_h - h) / 2):int((blank_h + h) / 2), int((blank_w - w) / 2):int((blank_w + w) / 2), :] = image
        image = blank

    return image


def get_alpha_bbox(image):
    a = np.where(image[:, :, 3] != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def resize_to_bbox(image, bbox):
    return image[bbox[0]:bbox[1], bbox[2]:bbox[3]]


def get_translation_M(t_x, t_y, t_z):
    return np.array([[1, 0, 0, t_x],
                     [0, 1, 0, t_y],
                     [0, 0, 1, t_z],
                     [0, 0, 0, 1]])


def get_rotation_M(r_x, r_y, r_z):
    sin_rx, cos_rx = np.sin(r_x), np.cos(r_x)
    sin_ry, cos_ry = np.sin(r_y), np.cos(r_y)
    sin_rz, cos_rz = np.sin(r_z), np.cos(r_z)
    # get the rotation matrix on x axis
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, sin_rx, 0],
                     [0, -sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz, cos_rz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # compute the full rotation matrix
    return np.dot(np.dot(R_Mx, R_My), R_Mz)


def get_scale_M(sc_x, sc_y, sc_z):
    return np.array([[sc_x, 0, 0, 0],
                     [0, sc_y, 0, 0],
                     [0, 0, sc_z, 0],
                     [0, 0, 0, 1]])


def get_projective_Ms(w, h, f):
    # set the image from cartesian to projective dimension
    H_M = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 1],
                    [0, 0, 1]])
    # set the image projective to carrtesian dimension
    Hp_M = np.array([[f, 0, w / 2, 0],
                     [0, f, h / 2, 0],
                     [0, 0, 1, 0]])

    return H_M, Hp_M


def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1)
              ):
    # get the height and the width of the image

    if len(image.shape) > 2 and image.shape[2] < 4:
        image = add_alpha(image)

    image = const_resize(image, np.max(image.shape[:2]) * 3 * np.max(scaling))

    h, w = image.shape[:2]

    # get the values on each axis
    t_x, t_y, t_z = translation
    theta_rx, theta_ry, theta_rz = np.deg2rad(rotation)
    sc_x, sc_y, sc_z = scaling

    # compute the focal length
    f = np.sqrt(h ** 2 + w ** 2)
    if np.sin(theta_rz) != 0:
        f /= 2 * np.sin(theta_rz)

    t_z = (f - t_z) / sc_z ** 2

    H_M, Hp_M = get_projective_Ms(w, h, f)

    # translation matrix to translate the image
    T_M = get_translation_M(t_x, t_y, t_z)

    # calculate cos and sin of angles
    R_M = get_rotation_M(theta_rx, theta_ry, theta_rz)

    # get the scaling matrix
    Sc_M = get_scale_M(sc_x, sc_y, sc_z)

    # compute the full transform matrix
    M = np.dot(R_M, H_M)
    M = np.dot(Sc_M, M)
    M = np.dot(T_M, M)
    M = np.dot(Hp_M, M)
    # apply the transformation
    image = cv2.warpPerspective(image, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

    bbox = get_alpha_bbox(image)

    image = resize_to_bbox(image, bbox)

    return image
