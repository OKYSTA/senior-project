from glob import glob
from importlib.resources import path
from nturl2path import pathname2url
import shutil
from tkinter.ttk import LabeledScale
import cv2
import os
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# パラメータを決める
def search_eps(dataset):
    nearest_neighbors = NearestNeighbors(n_neighbors=5)
    nearest_neighbors.fit(dataset)
    distances, indices = nearest_neighbors.kneighbors(dataset)
    distances = np.sort(distances, axis=0)[:, 1]
    print(distances)
    plt.plot(distances)
    plt.show()

# dbscanを実行する
def do_dbscan(dataset, param=6000):
    # モデルの作成
    model = DBSCAN(eps=param, min_samples=5).fit(dataset)
    labels = model.labels_
    fts = model.n_features_in_

    # print(labels)
    
    return model

# 結果からディレクトリごとに画像を分ける
def make_dir(dbscan, path):
    new_dir_path = '../result/dbscan/'
    for label in (dbscan.labels_):
        os.makedirs(new_dir_path + 'cluster{}'.format(label), exist_ok=True)

    for label, p in zip(dbscan.labels_, path):
        shutil.copy(p, new_dir_path + 'cluster{}/{}'.format(label, p.split('\\')[-1]))

# 画像をnumpy配列で読み込み、変形
impathlist = glob("convert-jpg/*")
features = np.array([cv2.resize(cv2.imread(p), (64, 64), interpolation=cv2.INTER_CUBIC) for p in impathlist])
features = features.reshape(features.shape[0], -1)

# search_eps(features)
# model = do_dbscan(features, 5200)
# make_dir(model, impathlist)
