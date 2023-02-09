import pyrealsense2 as rs
import numpy as np
import os
import cv2

# カメラの設定
conf = rs.config()
# RGB
conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 距離
conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

"""
conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
"""

# conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# decimarion_filterのパラメータ
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 1)

# spatial_filterのパラメータ
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
spatial.set_option(rs.option.filter_smooth_delta, 50)

# hole_filling_filterのパラメータ
hole_filling = rs.hole_filling_filter()

# disparity
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# alignモジュールの定義
align_to = rs.stream.color
align = rs.align(align_to)

cnt = 0

key_count = 110
DIR_PATH = '../datasets/captuers'
BASE_NAME = 'command_capture'

# stream開始
pipe = rs.pipeline()
profile = pipe.start(conf)

colorizer = rs.colorizer()

def capture(dir_path, basename, color_image, depth_image, key_count, ext='jpg'):
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    cv2.imwrite('{}_color_{}.{}'.format(base_path, key_count, ext), color_image)
    cv2.imwrite('{}_depth_{}.{}'.format(base_path, key_count, ext), depth_image)

def create_ply(key_count, colorized):
    ply = rs.save_to_ply(DIR_PATH + '/ply/{}_{}.ply'.format(BASE_NAME,key_count))
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    ply.process(colorized)


def affin(dir_path, basename, color_image, key_count, ext='jpg'):
    os.makedirs(dir_path, exist_ok=True)
    basepath = os.path.join(dir_path, basename)
    img = color_image
    # 画像サイズ
    height = img.shape[0]
    width = img.shape[1]

    # 座標指定([左上,左下,右下,右上])
    src_pts = np.array([[int(width*3/8), int(height*3/8)], [int(width/4), int(height*5/8)], [int(width*3/4), int(height*5/8)], [int(width*5/8), int(height*3/8)]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [0, 384], [384, 384], [384, 0]], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    perspective_img = cv2.warpPerspective(img, mat,(384, 384))

    cv2.imwrite('{}_affin_{}.{}'.format(basepath, key_count, ext), perspective_img)
try:
    while True:

        # frame処理で合わせる
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)
        aligned_frames = align.process(frames)
        depth_frame =  aligned_frames.get_depth_frame()
        color_frame =  aligned_frames.get_color_frame()

        """
        # frameデータを取得
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        """

        # filterをかける
        filter_frame = decimate.process(depth_frame)
        filter_frame = depth_to_disparity.process(filter_frame)
        filter_frame = spatial.process(filter_frame)
        filter_frame = disparity_to_depth.process(filter_frame)
        filter_frame = hole_filling.process(filter_frame)
        result_frame = filter_frame.as_depth_frame()

        # データを取得
        # color_data = color_frame.get_data()
        # depth_data = depth_frame.get_data()

        # データ配列を取得
        depth_image = np.asanyarray(depth_frame.get_data())

        # 画像データに変換
        color_image = np.asanyarray(color_frame.get_data())
        # 距離情報をカラースケール画像に変換する
        # depth_color_frame = rs.colorizer().colorize(depth_frame.get_data())
        depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # depth_array = np.asanyarray(depth_color_frame.get_data())

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)
        #お好みの画像保存処理
        if key == ord('c'):
            key_count += 1
            capture(DIR_PATH + '/original', BASE_NAME, color_image, depth_color_frame, key_count)
            create_ply(key_count, colorized)
            # affin(DIR_PATH + '/affin', BASE_NAME, color_image, key_count)
            cv2.destroyWindow('RealSense')
        elif key == ord('q'):
            cv2.destroyWindow('RealSense')
            break

finally:
    pipe.stop()