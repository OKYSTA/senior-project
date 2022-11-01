import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans


# 画像をuint8型として読み込む
image = Image.open('affin_start.png')
#image.show()

# numpy配列に変換しShapeを確認
image_np = np.array(image)
print('Image shape', image_np.shape)

# 第4番目のチャンネルは透過度（アルファ）なので除外する
# この画像では透過度は使われていない。使われていても、
# 除外した方が色が濃いのでより効果がはっきりするかもしれない。
original_image = image_np[:,:,:3]

# 画像を表示
plt.imshow(original_image)
plt.show()

# 各ピクセルを3次元（あるいは4次元）の特徴量として扱うため平坦化する
(h,w,c) = original_image.shape
data_points = original_image.reshape(h*w, c)
print('平坦化後', data_points.shape)

# K-Means法でクラスタリング(代表的な色を決める)
kmeans_model = KMeans(n_clusters=6) # クラスタ数を指定
cluster_labels = kmeans_model.fit_predict(data_points)

print('クラスターに振り分けされたデータ')
print(cluster_labels)

labels_count = Counter(cluster_labels)

print('クラスター毎のデータ数')
print(labels_count)

print('クラスター毎の代表的な色')
print(kmeans_model.cluster_centers_)

# 整数に変換
rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

print('クラスター毎の代表的な色')
print(rgb_cols)

# クラスターのラベルを画像のピクセルごとにRGBに変換
clustered_colors = rgb_cols[cluster_labels]
clustered_image = np.reshape(clustered_colors, (h, w, c))
print(clustered_image.shape)

# オリジナルとクラスタリングされた画像を表示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.imshow(original_image)
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(clustered_image)
ax2.set_title('Clustered Image')
ax2.set_axis_off()
plt.show()