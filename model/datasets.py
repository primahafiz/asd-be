from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class Datasets(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = ToTensor()

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return len(self.data)