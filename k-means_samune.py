from glob import glob
from msilib.schema import Component
import shutil
import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

CLUSTERS = int(input('input num of clusters ->'))
COMPONENTS = int(input('input num of components ->'))
OUTPUT_DIR = "../result/k-means/test"
 
# 画像をnumpy配列で読み込み、変形
impathlist = glob("../convert-jpg/*")
features = np.array([cv2.resize(cv2.imread(p), (64, 64), cv2.INTER_CUBIC) for p in impathlist])
features = features.reshape(features.shape[0], -1)

pca = PCA(n_components=22)
components = pca.fit_transform(features)
print("PCA累積寄与率: {}".format(sum(pca.explained_variance_ratio_)))
 
# モデルの作成
# model = KMeans(n_clusters=CLUSTERS).fit(features)
model = KMeans(n_clusters=CLUSTERS).fit(components)
 
# クラスタ数を変更して試したいので古い出力結果は消す
for i in range(model.n_clusters):
    cluster_dir = OUTPUT_DIR + "{}/cluster{}".format(CLUSTERS, i)
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
# 結果をクラスタごとにディレクトリに保存
for label, p in zip(model.labels_, impathlist):
    shutil.copyfile(p, OUTPUT_DIR + '{}/cluster{}/{}'.format(CLUSTERS, label, p.split('\\')[-1]))