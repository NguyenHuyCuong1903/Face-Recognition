import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from FaceNetModel import FaceNetModel
import joblib
from torchvision import transforms
import torch
from PIL import Image
import face_detection 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_facenet = FaceNetModel()
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)
model_classification = joblib.load('classification_model.pkl')

def load_model(model_path):
    load_weights = torch.load(model_path)
    model_facenet.load_state_dict(load_weights)

load_model('./facenet_model.pth')
model_facenet.to(device).eval()

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.classification_label = QLabel(self)
        self.layout.addWidget(self.classification_label)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.classify_button = QPushButton("Classify", self)
        self.classify_button.clicked.connect(self.classify_image)
        self.layout.addWidget(self.classify_button)

        self.central_widget.setLayout(self.layout)

        self.loaded_image = None

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif);;All Files (*)", options=options)
        
        if image_path:
            self.loaded_image = cv2.imread(image_path)
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaledToWidth(400))
            self.image_label.setFixedSize(500, 300)
            self.classification_label.setText("")

    def classify_image(self):
        if self.loaded_image is not None:
            detections = detector.detect(self.loaded_image)
            for detection in detections:
                x, y, z, t, score = map(int, detection)
                img = self.loaded_image[y:t, x:z]
                img =  cv2.resize(img, (160,160), cv2.INTER_LINEAR)
                img = transforms.ToTensor()(img).unsqueeze_(0)
                embedding = model_facenet(img.to(device)).cpu().detach().numpy()
                recognized_name = model_classification.predict(embedding)[0]
                prob = np.max(model_classification.predict_proba(embedding))
                if prob < 0.80:
                    recognized_name = 'unknow'
                recognized_name = recognized_name + ' probability:' + str(round(prob,2))
                self.classification_label.setText(f"Classification: {recognized_name}")
def main():
    
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()