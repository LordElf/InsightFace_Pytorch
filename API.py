import cv2
import argparse
from pathlib import Path
from PIL import Image
from torch.nn.modules.module import T
from mtcnn import MTCNN
from datetime import datetime
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
args = parser.parse_args()
data_path = Path('InsightFace_Pytorch/data')
# inital camera

def add_to_facebank(name:str='unknown', imglst=[]):
    mtcnn = MTCNN()
    save_path = data_path/'facebank'/name
    if not save_path.exists():
        save_path.mkdir()
    try:
        for img in imglst:
            warped_face = np.array(mtcnn.align(img))[...,::-1]
            cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
        print('data added')
    except:
        print('no face captured, please try another picture or retake a photo')

def take_pics(name:str='unknown'):
    if name == 'unknown':
        return False
    else: 
        mtcnn = MTCNN()
        save_path = data_path/'facebank'/name
        if not save_path.exists():
            save_path.mkdir()
        mtcnn = MTCNN()
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)

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
                    input = cv2.waitKey(1)&0xFF 
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

        cv2.destroyAllWindows()
        return True

def verify_face(save:bool=False, threshold=1.54, update:bool=True, tta:bool=True, is_score:bool=False):
    mtcnn = MTCNN()
    conf = get_config(False)
    learner = face_learner(conf, True)
    learner.threshold = threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    if save:
        video_writer = cv2.VideoWriter(conf.data_path/'recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...

    corr_times = 0
    name = ''
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    while cap.isOpened():
        isSuccess,frame = cap.read()
        print(corr_times)
        if corr_times >= 5:
            cap.release()
            cv2.destroyAllWindows()
            print(1)
            print(name)
            return True, name
        if isSuccess:            
            try:                    
                # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets, tta)
                for idx,bbox in enumerate(bboxes):
                    name = names[results[idx] + 1]
                    if is_score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, name, frame)
                        print(name)
                if name != 'Unknown': 
                    corr_times += 1
                # cv2.imshow("faceDec", frame)
            except:
                print('detect error')    

            if cv2.waitKey(1)&0xFF == ord('q'):
                break    

    if save:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    return False, None 