from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
centers = [[1, 0.5], [2, 2], [1, -1]] # 答えは3つのクラスターに分かれる
stds = [0.1, 0.4, 0.3] # それぞれのクラスターの標準偏差 (ばらつき具合)
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=stds, random_state=0)
fig = plt.figure(figsize=(10, 10))
# sns.scatterplot(X,hue=["cluster-{}".format(x) for x in labels_true])
# hue=["cluster-{}".format(x) for x in labels]
#sns.scatterplot(X[:,0], X[:,1])
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=10).fit(X)
labels = db.labels_
fig = plt.figure(figsize=(10, 10))
#sns.scatterplot(X[:,0], X[:,1])

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=.5, wspace=.2)
i = 1
for x in range(10, 0, -1):
    eps = 1/(11-x)
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    print(eps)
    ax = fig.add_subplot(2, 5, i)
    ax.text(1, 4, "eps = {}".format(round(eps, 1)), fontsize=25, ha="center")
    #sns.scatterplot(X[:,0], X[:,1])
    
    i += 1

db = DBSCAN(eps=0.178, min_samples=10).fit(X)
labels = db.labels_
fig = plt.figure(figsize=(10, 10))
#sns.scatterplot(X[:,0], X[:,1])