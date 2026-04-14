import os
import urllib.request
import tarfile
import subprocess

FILE = "cityscapes"
URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{FILE}.tar.gz"
TAR_FILE = f"./datasets/{FILE}.tar.gz"
TARGET_DIR = f"./datasets/{FILE}/"

# 创建目录
os.makedirs(TARGET_DIR, exist_ok=True)

# 下载数据集
print(f"Downloading {URL}...")
try:
    urllib.request.urlretrieve(URL, TAR_FILE)
    print("Download completed!")
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

# 解压
print(f"Extracting to {TARGET_DIR}...")
with tarfile.open(TAR_FILE, 'r:gz') as tar:
    tar.extractall(path="./datasets/")

# 删除压缩包
os.remove(TAR_FILE)
print("Tar file removed!")

# 生成列表文件
print("Generating dataset lists...")
train_dir = os.path.join(TARGET_DIR, "train")
val_dir = os.path.join(TARGET_DIR, "val")

# 获取所有 jpg 文件
train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
val_files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')])

# 写入列表文件
with open('train_list.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open('val_list.txt', 'w') as f:
    for file in val_files:
        f.write(file + '\n')

print(f"train_list.txt: {len(train_files)} images")
print(f"val_list.txt: {len(val_files)} images")
print("下载完成，已生成列表，可以训练")
