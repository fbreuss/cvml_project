import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from ros_numpy.occupancy_grid import numpy_to_occupancy_grid
from tf.transformations import euler_from_quaternion


# image stuff
output_path = "output/"
img_name = 'stuttgart_01_000000_003715_leftImg8bit.png'
# img_name = 'frame_0_input.png'
# img_name = 'frame_0_confidence.png'
# img_name = 'frame_0_prediction.png'
# img_name = 'dummy.png'



# ROS stuff
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import Image, CameraInfo
# Publishers
pub_map = rospy.Publisher('/map_terrain2', OccupancyGrid, queue_size=1)
pub_map_info = rospy.Publisher('/map_terrain_info', MapMetaData, queue_size=1)


############################################# PARAMETERS
# camera info
focal_length = 0.008
pixel_size = 0.00000586
img_width = 640
img_height = 320
camera_width = 1920
camera_height = 1200
# position relative to base_link
camera_translation = [0.8031200000000001, 0.1, 0.9505]
# TODO what happens with other rotational values?
camera_rotation = [0.5, -0.5, 0.5, -0.5]

camera_translation_height = [0, 0, camera_translation[2]]

# rotation around: x, y, z
(cam_roll, cam_pitch, cam_yaw) = euler_from_quaternion(
    [camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]])

# map info
TARGET_W_m, TARGET_H_m = 20, 15
RESOLUTION = 0.05    # [m / cells]

TARGET_W, TARGET_H = int(TARGET_W_m/RESOLUTION), int(TARGET_H_m/RESOLUTION)

image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)

import os
if not os.path.exists(output_path):
    os.makedirs(output_path)


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
    print("input", x, y)
    x = x / img_width * camera_width
    y = y / img_height * camera_height

    print("camera", x, y)
    # treat as if middle is zero
    x = x - camera_width / 2
    y = y - camera_height / 2
    print("zeroed", x, y)

    x *= pixel_size
    y *= pixel_size

    pixel = [x, y, focal_length]
    print("pixel 2D", pixel)
    # transfer pixel coordinate to 3D coordinate
    pixel = transfer_data(cam_roll, cam_pitch, cam_yaw, camera_translation_height, pixel)
    print("pixel 3D", pixel)

    # project 3D pixel coordinate to ground (similar triangles, long1/short1 = long2/short2 -> long1 = short1 * long2/short2)
    h = camera_translation_height[2]   # total height of center (long side)
    dh = h - pixel[2]   # height difference of center to pixel (short side)

    if dh <= 0:
        print("THIS SHOULD NOT HAPPEN!", dh)
        exit(1)

    X = pixel[0] * h / dh
    Y = pixel[1] * h / dh

    # X = X / camera_width * img_width
    # Y = Y / camera_height * img_height

    return [X, Y]


def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


if __name__ == '__main__':
    rospy.init_node("ipm", anonymous=True)
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
        print("p", p)
        t.append([p[0] / RESOLUTION, p[1] / RESOLUTION])
    t = np.asarray(t, dtype=np.float32)
    # preprocess target points to center the projected image inside the map
    minLeft = np.min(t, axis=0)[0]
    for p in t:
        # move points (-> map) towards center
        p[1] += TARGET_H/2
        # move points towards left edge of map (to minimize zero values)
        p[0] -= minLeft
    print("t", t)


    # Warp the image
    stt = time.time()
    warped2 = ipm_from_opencv(image, s, t)
    dt = time.time() - stt
    print("warp time", dt)


    # publish ros topic
    stt = time.time()
    occ_map = numpy_to_occupancy_grid(warped2.astype(np.int8))
    dt2 = time.time() - stt
    print("convert 2 map time", dt2)

    occ_map.header.stamp = rospy.Time.now()
    occ_map.header.frame_id = "base_link"
    occ_map.info.resolution = RESOLUTION
    occ_map.info.width = TARGET_W
    occ_map.info.height = TARGET_H
    occ_map.info.origin = Pose(Point(camera_translation[0] + minLeft*RESOLUTION, camera_translation[1] - TARGET_H_m/2, 0),
                                    Quaternion(0, 0, 0, 1))


    # Draw results
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(img_name + "\ntimes: warp {:.6f}s\nconvert2map {:.6f}s".format(dt, dt2))
    ax[0].imshow(image)
    ax[0].set_title('Front View')
    ax[0].set_xlabel('width')
    ax[0].set_ylabel('height')
    ax[1].imshow(warped2)
    ax[1].set_title('Generated Map')
    ax[1].set_xlabel('TARGET_W')
    ax[1].set_ylabel('TARGET_H -> forward')
    plt.tight_layout()
    plt.savefig(output_path + img_name)
    plt.show()

    # while True:
    pub_map.publish(occ_map)
    #     rospy.sleep(0.5)
    # pub_map_info.publish(terrain_msgs.info)
