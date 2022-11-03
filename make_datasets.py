from glob import glob
from msilib.schema import Component
import shutil
import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

IMPATH_LIST = ""