import cv2 as cv2
import os
import numpy as np
import argparse

def extact_frames(video_path="data/train.mp4" ,save_path="data/frames_train", ext = '.jpg'):
    success = True
    img = None
    cnt = 0
    vidcap = cv2.VideoCapture(video_path)
    while success:
        print(cnt)
        success, img = vidcap.read()
        cv2.imwrite(os.path.join(save_path,str(cnt)+ext),img)
        cnt += 1

# This code is from https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
def dense_optical_flow(prev_frame,next_frame):
    hsv = np.zeros_like(prev_frame)
    prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_RGB2GRAY)
    next_frame = cv2.cvtColor(next_frame,cv2.COLOR_RGB2GRAY)
    hsv[..., 1] = 255
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', action='store', type=str, default="data/frames_test",
                        help='Directory containing the training images')
    parser.add_argument('--video_path', action='store', type=str, default="data/test.mp4",
                        help='Directroy containing the test images')

    args = parser.parse_args()
    extact_frames(video_path=args.video_path,save_path=args.save_path)