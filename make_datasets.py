from glob import glob
from msilib.schema import Component
import shutil
import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from random import randint

INPUTDIR_PATH = "../datasets/convert_jpg/sets/set"
OUTPUTDIR_PATH = "../datasets/choice_jpg{}/set{}"
SETS_NUMBER = 11
GET_NUMBER = int(input('how many photos chice? ->'))

def set_labels():
    labels = []
    for i in range(SETS_NUMBER):
        labels.append(i)
    return labels

def make_dir():
    for i in range(SETS_NUMBER):
        dir_name = OUTPUTDIR_PATH.format(GET_NUMBER, i)
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    
    return 

def copy_imgaes(labels):
    for i in range(SETS_NUMBER):
        inpath_list = glob(INPUTDIR_PATH + "{}/*".format(i))
        count = 0
        for j in range(GET_NUMBER):
            limit = len(inpath_list) - 1
            target = inpath_list.pop(randint(0, limit))
            shutil.copyfile(target, OUTPUTDIR_PATH.format(GET_NUMBER, labels[i]) + "/{}".format(target.split("\\")[-1]))
    return

label = set_labels()
make_dir()
copy_imgaes(label)