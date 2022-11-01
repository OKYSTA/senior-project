import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_img(path, s, gray=False):
    img_bgr = cv2.imread(path + '/' + s)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if gray:
        return img_gray
    else:
        return img_rgb

def read_imges(name, path='image/', gray=False):
    path = path + name

    list1 = os.listdir(path)
    tmp = np.array([read_img(path, s, gray=gray) for s in list1])
    return tmp

def make_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    kernel = np.ones((3, 3), np.uint8)
    ret, thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, kernel, iterations = 6)
    thresh = cv2.erode(thresh, kernel, iterations = 2)
    mask = cv2.bitwise_not(thresh)
    return mask

def calc_roundness(rgb):
    thresh = make_mask(rgb)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    roundness = 4*np.pi*area / perimeter**2
    return roundness

def mean_col(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = make_mask(rgb)
    
    r = np.sum(rgb[:,:,0]*mask)/np.sum(mask==255)
    g = np.sum(rgb[:,:,1]*mask)/np.sum(mask==255)
    b = np.sum(rgb[:,:,2]*mask)/np.sum(mask==255)
    
    return np.array([r, g, b])