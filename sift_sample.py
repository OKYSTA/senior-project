import cv2
import numpy as np

img = cv2.imread('../convert-jpg/IMG_0210.jpg')
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
cv2.imwrite("../result/sift/sift_img.jpg", img_sift)