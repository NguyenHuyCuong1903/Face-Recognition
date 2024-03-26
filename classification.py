import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import os
from FaceNetModel import FaceNetModel
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Khai báo các biến và đường dẫn
model_path = 'facenet_model.pth'  # Đường dẫn đến tệp .pth của mô hình
data_dir = './Data/test'  # Đường dẫn đến thư mục dữ liệu

# Chuẩn bị dữ liệu
data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


model_facenet = FaceNetModel()
# Load mô hình đã được huấn luyện từ tệp .pth
def load_model(model_path):
    load_weights = torch.load(model_path)
    model_facenet.load_state_dict(load_weights)

load_model(model_path)
model_facenet.to(device).eval()                                              # svm   cay  bayes 
# model_facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()  99.42% 92.28% 99.11%
# model_facenet = InceptionResnetV1(pretrained='casia-webface').to(device).eval()  99.23% 84.34% 99.42%
# Trích xuất biểu diễn nhúng cho dữ liệu huấn luyện                                 99.85   95.99%  99.81%
def extract_embeddings(model, phase = 'train'):
    train_embeddings = []
    train_labels = []
    root_path = f'./Data/{phase}'
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            img = Image.open(file_path)
            img = data_transform(img).unsqueeze_(0)
            embeddings = model(img.to(device))
            embeddings = embeddings.cpu().detach().numpy()[0]
            train_embeddings.append(embeddings)
            train_labels.append(folder)
    return train_embeddings, train_labels


train_embeddings, train_labels = extract_embeddings(model_facenet, 'train')
test_embeddings, test_labels = extract_embeddings(model_facenet, 'test')

# Khởi tạo mô hình SVM
svm ty=True)

# Huấn luyện mô hình SVM trên biểu diễn nhúng của dữ liệu huấn luyện
svm_classifier.fit(train_embeddings, train_labels)

# Sử dụng mô hình SVM để dự đoán nhãn cho dữ liệu thử nghiệm
predicted_labels = svm_classifier.predict(test_embeddings)

# Đánh giá mô hình SVM
accuracy = accuracy_score(test_labels, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')
joblib.dump(svm_classifier, 'classification_model.pkl')