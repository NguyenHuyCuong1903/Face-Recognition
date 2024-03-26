import random
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [label for data, label in self.dataset.imgs]

    def __getitem__(self, index):
        anchor, anchor_label = self.dataset[index]
        positive_index = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label])
        negative_index = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])
        positive, _ = self.dataset[positive_index]
        negative, _ = self.dataset[negative_index]
        return anchor, positive, negative

    def __len__(self):
        return len(self.dataset)