import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 - 1.0
        
        # 优化维度分格，适应各种尺寸的图片
        mid_w = image.shape[2] // 2  # 宽度的中点
        image_rgb = image[:, :, :mid_w]      # 左半部分
        image_semantic = image[:, :, mid_w:] # 右半部分
        
        return image_rgb, image_semantic
