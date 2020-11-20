from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import imageio
import cv2
import os


class AFADDataset(Dataset):
    def __init__(self, data_root, index_root, input_size, augment):
        self.data = []
        self.data_root = data_root
        self.input_size = input_size
        self.augment = augment

        with open(index_root, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                path = os.path.join(self.data_root, line[2:-1])
                if os.path.exists(path) and 'Thumbs' not in path:
                    self.data.append(line[2:-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.resize(imageio.imread(os.path.join(self.data_root, self.data[idx])), (self.input_size, self.input_size))
        image = preprocess(image, self.input_size, self.augment)
        gender = np.array(int(self.data[idx][5]) - 1, dtype=np.int64)
        age = np.array(int(self.data[idx][:2]), dtype=np.float32)
        return image, gender, age


def preprocess(image, input_size, augmentation=True):
    if augmentation:
        crop_transform = transforms.Compose([
            transforms.Resize(input_size // 4 * 5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(input_size),
            transforms.RandomRotation(10)])
    else:
        crop_transform = transforms.CenterCrop(input_size)

    result = transforms.Compose([
        transforms.ToPILImage(),
        crop_transform,
        transforms.ToTensor(),
    ])(image)
    return result
