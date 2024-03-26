import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from FaceNetModel import FaceNetModel
from TripletDataset import TripletDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Chuẩn bị dữ liệu
data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='./Data/train', transform=data_transform)
val_dataset = ImageFolder(root='./Data/test', transform=data_transform)

triplet_dataset_train = TripletDataset(train_dataset)
triplet_loader_train = DataLoader(triplet_dataset_train, batch_size=16, shuffle=True)
triplet_dataset_val = TripletDataset(val_dataset)
triplet_loader_val = DataLoader(triplet_dataset_val, batch_size=8, shuffle=True)

# Khởi tạo mô hình FaceNet-PyTorch và hàm mất mát triplet loss
model = FaceNetModel()
model.to(device)

# Định nghĩa hàm tối ưu hóa
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
L_train = []
L_val = []
def train(model, num_epochs, optimizer, triploss):
    # Huấn luyện mô hình sử dụng triplet loss
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        print(f'Train epoch {epoch + 1}')
        for i, (anchor, positive, negative) in enumerate(triplet_loader_train):
            optimizer.zero_grad()
            anchor_embedding = model(anchor.to(device))
            positive_embedding = model(positive.to(device))
            negative_embedding = model(negative.to(device))
            loss = triploss(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            print(f'    Epoch [{epoch + 1}/{num_epochs}], Batchsize: {i+1}, Total Loss: {training_loss}')
        # phase valid
        torch.cuda.empty_cache()
        model.eval()
        valid_loss = 0.0
        for j, (anchor, positive, negative) in enumerate(triplet_loader_val):
            anchor_embedding = model(anchor.to(device))
            positive_embedding = model(positive.to(device))
            negative_embedding = model(negative.to(device))
            loss = triploss(anchor_embedding, positive_embedding, negative_embedding)
            valid_loss += loss.item()
        L_train.append(round(training_loss / (i + 1),4))
        L_val.append(round(valid_loss / (j + 1),4))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_loss / (i + 1):.4f}, Validation Loss: {valid_loss / (j + 1):.4f}')

    print('Training complete')
    torch.save(model.state_dict(), 'facenet_model.pth')

if __name__ == '__main__':
    num_epochs = 10
    train(model, num_epochs, optimizer, triplet_loss)
    print(L_train)
    print(L_val)