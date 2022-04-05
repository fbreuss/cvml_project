import numpy as np
import cv2
import time
from numba import jit, cuda

# TODO sudo pip install numba
# TODO sudo pip install numpy==1.21

@jit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@jit(nopython=True)
def confidence_one_hot(image):
    x = np_apply_along_axis(np.max, -1, image)
    return x

@jit(nopython=True)
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x

@jit(nopython=True)
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

@jit(nopython=True)
def confidence_image(image, threshold, label_values_drivable):
    if threshold is None:
        threshold = 0.76
    confidence_img = image
    threshold_img = image
    confidence_img = confidence_one_hot(confidence_img)
    threshold_img = reverse_one_hot(threshold_img)
    idxs = confidence_img <= threshold
    threshold_img[idxs] = 0
    threshold_img = colour_code_segmentation(threshold_img, label_values_drivable)
    return threshold_img


img_name = 'frame_0_prediction.png'
prediction_image = cv2.imread(img_name)
prediction_image = cv2.resize(prediction_image, (640, 320), interpolation=cv2.INTER_LINEAR)
prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_RGB2BGR)
label_values_drivable = [[0, 0, 0], [0, 255, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]

n = 1
stt = time.time()
for i in range(n):
    confidence_vis_image = confidence_image(prediction_image, 0.76, label_values_drivable)
dt = (time.time() - stt) / n
print("warp time", dt)

# CPU 0.006309107065200805
# GPU