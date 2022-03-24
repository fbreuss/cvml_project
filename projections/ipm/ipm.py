import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import perspective, Plane, load_camera_params, bilinear_sampler, warped

from tf.transformations import euler_from_quaternion

image = cv2.cvtColor(cv2.imread('frame_0_input.png'), cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(cv2.imread('dummy.png'), cv2.COLOR_BGR2RGB)
interpolation_fn = bilinear_sampler  # or warped


# camera info
focal_length = 0.008
pixel_size = 0.00000586
img_width = 640
img_height = 320
# position relative to base_link
camera_translation = [0.8031200000000001, 0.1, 0.9505]
camera_translation = [0, 0, 5]
# TODO what happens with other rotational values?
camera_rotation = [0.5, -0.5, 0.5, -0.5]

# rotation around: x, y, z
(cam_roll, cam_pitch, cam_yaw) = euler_from_quaternion(
    [camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]])

# map info
TARGET_W, TARGET_H = 300, 200
RESOLUTION = 100    # [cells / m]


def transfer_data(x_angle, y_angle, z_angle, translation, point):
    """
    function to translate and rotate data points, Note: cardan Rotation Matrix
    :param x_angle: rotation for x axis in radiant
    :param y_angle: rotation for y axis in radiant
    :param z_angle: rotation for z axis in radiant
    :param translation: translation (x;y;z)
    :param point: one or more dimension (x;y;z)
    :return: transformated data points (x;y;z)
    """
    # cardan Roation Matrix - von Camera in World Frame
    Rz = np.array([[np.cos(z_angle), -np.sin(z_angle), 0], [np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(x_angle), -np.sin(x_angle)], [0, np.sin(x_angle), np.cos(x_angle)]])
    A = np.matmul(np.matmul(Rz, Ry), Rx)
    return np.dot(A, point) + translation


def image_to_3D_groundplane(x, y):
    # x,y ... 2D coordinate in image [pixel]
    # X,Y ... 3D coordinate on ground (Z = 0) relative to camera center (inclusive translation & rotation) [m]

    print("")
    print("c", x, y)
    # treat as if middle is zero
    x = x - img_width / 2
    y = y - img_height / 2
    print("c", x, y)

    x *= pixel_size
    y *= pixel_size

    pixel = [x, y, focal_length]
    print("pixel 2D", pixel)
    # transfer pixel coordinate to 3D coordinate
    pixel = transfer_data(cam_roll, cam_pitch, cam_yaw, camera_translation, pixel)
    print("pixel 3D", pixel)

    # project 3D pixel coordinate to ground (similar triangles, long1/short1 = long2/short2 -> long1 = short1 * long2/short2)
    h = camera_translation[2]   # total height of center (long side)
    dh = h - pixel[2]   # height difference of center to pixel (short side)

    if dh <= 0:
        print("THIS SHOULD NOT HAPPEN!", dh)
        exit(1)

    X = pixel[0] * h / dh
    Y = pixel[1] * h / dh

    return [X, Y]


def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    print(M)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


if __name__ == '__main__':
    ################
    # OpenCV
    ################
    # Vertices coordinates in the source image, tlr
    s = np.array([[0, img_height/2+50],
                  [img_width, img_height/2+50],
                  [0, img_height],
                  [img_width, img_height]], dtype=np.float32)
    print("s", s)

    # draw points
    img_circles = image.copy()
    for item in s:
        img_circles = cv2.circle(img_circles, (int(item[0]), int(item[1])), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imshow("source image", img_circles)
    # cv2.waitKey()

    # Vertices coordinates in the destination image, tlr
    t = []
    for i in s:
        p = image_to_3D_groundplane(i[0], i[1])
        p[1] += TARGET_H/2
        t.append(p)
    t = np.asarray(t, dtype=np.float32)
    print("t", t)


    # Warp the image
    stt = time.time()
    warped2 = ipm_from_opencv(image, s, t)
    print("time", time.time() - stt)


    # Draw results
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[0].set_title('Front View')
    ax[0].set_xlabel('width')
    ax[0].set_ylabel('height')

    ax[1].imshow(warped2)
    ax[1].set_title('IPM from OpenCv')
    ax[1].set_xlabel('TARGET_W')
    ax[1].set_ylabel('TARGET_H')

    plt.tight_layout()
    plt.show()
