import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()
from pathlib import Path
data_path = Path('data')
save_path = data_path/'facebank'/args.name
if not save_path.exists():
    save_path.mkdir()

# 初始化摄像头
cap = cv2.VideoCapture(0)
# 我的摄像头默认像素640*480，可以根据摄像头素质调整分辨率
cap.set(3,1920)
cap.set(4,1080)
mtcnn = MTCNN()

if cap.isOpened():
    while True:
        # 采集一帧一帧的图像数据
        isSuccess,frame = cap.read()
        # 实时的将采集到的数据显示到界面上
        if isSuccess:
            frame_text = cv2.putText(frame,
                        'Press t to take a picture, q to quit.....',
                        (10,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        2,
                        (0,255,0),
                        3,
                        cv2.LINE_AA)
            cv2.imshow("My Capture",frame_text)
            # 实现按下“t”键拍照
            input =  cv2.waitKey(1)&0xFF
            if input == ord('t'): 
                p =  Image.fromarray(frame[...,::-1])
                try:
                    warped_face = np.array(mtcnn.align(p))[...,::-1]
                    cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
                    print('data added')
                except:
                    print('no face captured, please try another picture or retake a photo')

            elif input == ord('q'):
                break

# 释放摄像头资源
cap.release()