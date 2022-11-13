import cv2
import numpy as np

# 画像読込
# img = cv2.resize(cv2.imread('../datasets/pictures/affin.jpg'), (1024, 1024))
img = cv2.imread("../datasets/pictures/affin.jpg")

# 画像サイズ
height = img.shape[0]  # 高さ
width  = img.shape[1]  # 幅

def img_guruguru():
    #画像の回転行列
    rot_matrix = cv2.getRotationMatrix2D((width/2,height/2),  # 回転の中心座標
                                        45,                 # 回転する角度
                                        1,                  # 画像のスケール
                                        )

    # アフィン変換適用
    afin_img = cv2.warpAffine(img,               # 入力画像
                            rot_matrix,        # 行列
                            (width,height)     # 解像度
                            )

    # 画像表示
    cv2.imshow("img",afin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def syaei():
    src_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
    dst_pts = np.array([[300, 50], [0, 600], [1200, 600], [900, 50]], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print(mat)

    # perspective_img = cv2.warpPerspective(img, mat, (1300, 700))
    cv2.imwrite('../result/pictures/opencv_perspective.jpg', perspective_img)
    cv2.imshow("img", perspective_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

syaei()