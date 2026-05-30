import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
import torchvision.transforms as transforms
import os

batch_w = 600
batch_h = 400

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        self.transform = transforms.ToTensor()

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_tensor = self.transform(im)  # FloatTensor, CxHxW, [0,1]
        return img_tensor

    def __getitem__(self, index):
        low = self.load_images_transform(self.train_low_data_names[index])
        _, h, w = low.shape

        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))

        if self.task != 'test':
            low = low[:, h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        img_name = os.path.basename(self.train_low_data_names[index])

        return low, img_name

    def __len__(self):
        return self.count