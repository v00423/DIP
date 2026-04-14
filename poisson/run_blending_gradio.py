import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)  # 防止数值越界
    image = (image * 255).astype(np.uint8)
    return image

def align_tensor_to_target(output_tensor, target_tensor):
    """
    Align output tensor spatial size to target tensor spatial size.

    Args:
        output_tensor (torch.Tensor): shape [B, C, H, W]
        target_tensor (torch.Tensor): shape [B, C, Ht, Wt]
    """
    if output_tensor.shape[-2:] != target_tensor.shape[-2:]:
        output_tensor = F.interpolate(
            output_tensor,
            size=target_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    return output_tensor

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    b = inputs.size(0)
    n = min(num_images, b)

    for i in range(n):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # numpy层再做一次保险对齐，避免hstack报错
        h, w = target_img_np.shape[:2]
        if input_img_np.shape[:2] != (h, w):
            input_img_np = cv2.resize(input_img_np, (w, h), interpolation=cv2.INTER_LINEAR)
        if output_img_np.shape[:2] != (h, w):
            output_img_np = cv2.resize(output_img_np, (w, h), interpolation=cv2.INTER_LINEAR)

        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # 如果你用cv2看图颜色不对，可改成 cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        optimizer.zero_grad()

        outputs = model(image_rgb)
        outputs = align_tensor_to_target(outputs, image_semantic)  # 关键：对齐尺寸后再算loss

        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

        loss = criterion(outputs, image_semantic)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            outputs = model(image_rgb)
            outputs = align_tensor_to_target(outputs, image_semantic)  # 验证同样要对齐

            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    num_epochs = 300
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        validate(model, val_loader, criterion, device, epoch, num_epochs)

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
