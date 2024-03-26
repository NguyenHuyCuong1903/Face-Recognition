import cv2
import os
from facenet_pytorch import MTCNN
import face_detection
import torch
from matplotlib import pyplot as plt
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(thresholds=[0.7,0.8,0.8], device=device)
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)


def detect_face(root_path, target_path):
    for folder in os.listdir(root_path):
        folder_path = os.path.join(target_path, folder)
        os.makedirs(folder_path, exist_ok=True)

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        cnt = 0
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            img = cv2.imread(image_path)
            detections = detector.detect(img)
            for detection in detections:  # Assuming you are interested in the first set of face detections
                try:
                    x, y, z, t, score = map(int, detection)
                    img = img[y:t, x:z]
                    img =  cv2.resize(img, (160,160), cv2.INTER_LINEAR)
                    img = cv2.imwrite(os.path.join(target_path, folder, f'img{cnt}.png'), img)
                    cnt += 1
                except:
                    pass

if __name__ == '__main__':
    detect_face('./Img', './Img_detected')