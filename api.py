import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import joblib
from flask import Flask, request
import base64
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from FaceNetModel import FaceNetModel
import face_detection


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(thresholds=[0.7, 0.9, 0.9], device=device, min_face_size=20)
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
model_facenet = FaceNetModel()
# Load mô hình đã được huấn luyện từ tệp .pth
def load_model(model_path):
    load_weights = torch.load(model_path)
    model_facenet.load_state_dict(load_weights)

load_model('facenet_model.pth')
model_facenet.to(device).eval()

app = Flask(__name__)

# Load the model from a pickle file.
model_path = "classification_model.pkl"
model = joblib.load(open(model_path, 'rb'))

def convert_base64toimage(base64_image):
    try:
        base64_image = np.frombuffer(base64.b64decode(base64_image), dtype=np.uint8)
        base64_image = cv2.imdecode(base64_image, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return base64_image

# Define the API endpoint.
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request body.
    image_base64 = request.form.get("image")
    img = convert_base64toimage(image_base64)
    detections = detector.detect(img)
    if len(detections) > 0:
        x, y, z, t, score = map(int, detections[0])
        img = img[y:t, x:z] 
        img =  cv2.resize(img, (160,160), cv2.INTER_LINEAR)
        img = transforms.ToTensor()(img).unsqueeze_(0)
        query_embedding = model_facenet(img.to(device)).cpu().detach().numpy()
        recognized_name = model.predict(query_embedding)[0]
        prob = np.max(model.predict_proba(query_embedding))
        if prob < 0.8:
            return "Cannot recognite"
        else:
            return "Result: " + recognized_name + '\nProbability: ' + str(round(prob, 2))
    return "Cannot detect"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')
