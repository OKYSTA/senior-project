import cv2
import numpy as np
import glob

# 画像読込
# img = cv2.resize(cv2.imread('../datasets/pictures/affin.jpg'), (1024, 1024))

def onMouse(event, x, y, flag, params):
    raw_img = params["img"]
    wname = params["wname"]
    point_list = params["point_list"]
    point_num = params["point_num"]
    
    ## クリックイベント
    ### 左クリックでポイント追加
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(point_list) < point_num:
            point_list.append([x, y])
    
    ### 右クリックでポイント削除
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(point_list) > 0:
            point_list.pop(-1)

    ## レーダーの作成, 描画
    img = raw_img.copy()
    h, w = img.shape[0], img.shape[1]
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)

    ## 点, 線の描画
    for i in range(len(point_list)):
        cv2.circle(img, (point_list[i][0], point_list[i][1]), 3, (0, 0, 255), 3)
        if 0 < i:
            cv2.line(img, (point_list[i][0], point_list[i][1]),
                     (point_list[i-1][0], point_list[i-1][1]), (0, 255, 0), 2)
        if i == point_num-1:
            cv2.line(img, (point_list[i][0], point_list[i][1]),
                     (point_list[0][0], point_list[0][1]), (0, 255, 0), 2)
    
    if 0 < len(point_list) < point_num:
        cv2.line(img, (x, y),
                     (point_list[len(point_list)-1][0], point_list[len(point_list)-1][1]), (0, 255, 0), 2)
    """
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1000, int(1000*h/w))
    cv2.imshow(wname, img)
    """

def scale_box(img, width, height):
    """指定した大きさに収まるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)

    dst = cv2.resize(img, dsize=(nw, nh))

    return dst

def syaei(file, i):

    img = cv2.imread("{}".format(file))

    wname = "MouseEvent"
    point_list = []
    point_num = 8
    params = {
        "img": img,
        "wname": wname,
        "point_list": point_list,
        "point_num": point_num,
    }    
    
    # 画像サイズ
    height = img.shape[0]  # 高さ
    width  = img.shape[1]  # 幅

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 1000, int(1000*height/width))
    cv2.setMouseCallback("img", onMouse, params)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    top, bottom, left, right = point_list[0][1], point_list[2][1], point_list[0][0], point_list[1][0]
    crop_img = img[top:bottom, left:right]
    resize_img = scale_box(crop_img, 800, 600)
    cv2.imwrite('C:/Users/c0119059e1/grad/clustering/datasets/captures/original/crop/command_capture_affin_{}.jpg'.format(i), resize_img)

    ax, ay, bx, by, cx, cy, dx, dy = point_list[4][0], point_list[4][1], point_list[6][0], point_list[6][1], point_list[7][0], point_list[7][1], point_list[5][0], point_list[5][1]

    src_pts = np.array([[ax, ay], [bx, by], [cx, cy], [dx, dy]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [0, 640], [640, 640], [640, 0]], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print(mat)

    perspective_img = cv2.warpPerspective(img, mat, (640, 640))
    cv2.imwrite('C:/Users/c0119059e1/grad/clustering/datasets/captures/original/affin/command_capture_affin_{}.jpg'.format(i), perspective_img)
    """
    cv2.imshow("img", perspective_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

"""
files = glob.glob('C:/Users/c0119059e1/grad/clustering/datasets/captures/original/color/*')
i = 1
for file in files:
    syaei(file, i)
    i += 1
"""
file = "C:/Users/c0119059e1/grad/clustering/datasets/captures/original/color/command_capture_color_92.jpg"
i = 92
syaei(file, i)
