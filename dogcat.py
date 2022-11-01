# -*- coding: utf-8 -*-
import os
from secrets import choice
import sys
from tkinter.simpledialog import SimpleDialog
from tracemalloc import start
import cv2
import numpy as np
import random
from tqdm import tqdm

## 画像データのクラスIDとパスを取得
#
# @param dir_path 検索ディレクトリ
# @return data_sets [クラスID, 画像データのパス]のリスト
def getDataSet(dir_path):
    data_sets = []
    test_data_sets = []    

    sub_dirs = os.listdir(dir_path)
    firstloop = True
    for classId in sub_dirs:
        sub_dir_path = dir_path + '/' + classId
        img_files = os.listdir(sub_dir_path)
        for f in img_files:
            data_sets.append([classId, sub_dir_path + '/' + f])
        choice = 1500
        wide = len(data_sets) - 1
        for i in range(choice):
            test_data_sets.append(data_sets.pop(random.randint(0, wide)))
            wide -= 1
        if firstloop:
            start = len(data_sets)
            firstloop = False
    else:
        wide = len(data_sets) - 1
        for i in range(choice):
            test_data_sets.append(data_sets.pop(random.randint(start, wide)))
            wide -= 1


    return data_sets, test_data_sets

"""
main
"""
# 定数定義
GRAYSCALE = 0
# KAZE特徴量抽出器
detector = cv2.KAZE_create()

"""
train
"""
print("train start")
# 訓練データのパスを取得
train_set, test_set = getDataSet('dogs-cats/train')
# 辞書サイズ
dictionarySize = 2
# Bag Of Visual Words分類器
bowTrainer = cv2.BOWKMeansTrainer(dictionarySize)

# 各画像を分析
for i, (classId, data_path) in tqdm(enumerate(train_set)):
    # 進捗表示
    sys.stdout.write(".")
    # グレースケールで画像読み込み
    gray = cv2.imread(data_path, GRAYSCALE)
    # 特徴点とその特徴を計算
    keypoints, descriptors= detector.detectAndCompute(gray, None)
    # intからfloat32に変換
    descriptors = descriptors.astype(np.float32)
    # 特徴ベクトルをBag Of Visual Words分類器にセット
    bowTrainer.add(descriptors)

# Bag Of Visual Words分類器で特徴ベクトルを分類
codebook = bowTrainer.cluster()
# 訓練完了
print("train finish")

"""
test
"""
print("test start")
# テストデータのパス取得
#test_set = getDataSet("dogs-vs-cats/test1")

# KNNを使って総当たりでマッチング
matcher = cv2.BFMatcher()

# Bag Of Visual Words抽出器
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
# トレーニング結果をセット
bowExtractor.setVocabulary(codebook)

# 正しく学習できたか検証する
success = 0
count = 3000
for i, (classId, data_path) in enumerate(test_set):
    # グレースケールで読み込み
    gray = cv2.imread(data_path, GRAYSCALE)
    # 特徴点と特徴ベクトルを計算
    keypoints, descriptors= detector.detectAndCompute(gray, None)
    # intからfloat32に変換
    descriptors = descriptors.astype(np.float32)
    # Bag Of Visual Wordsの計算
    bowDescriptors = bowExtractor.compute(gray, keypoints)

    # 結果表示
    className = {"0": "cat",
                 "1": "dog"}

    actual = "???"    
    if bowDescriptors[0][0] > bowDescriptors[0][1]:
        actual = className["0"]
    elif bowDescriptors[0][0] < bowDescriptors[0][1]:
        actual = className["1"]

    result = ""
    if actual == "???":
        result = " => unknown."
    elif className[classId] == actual:
        result = " => success!!"
        success += 1
    else:
        result = " => fail"

    print("expected: ", className[classId], ", actual: ", actual, result)
result = int((success / count) * 100)
print(f"正解率{result}%")