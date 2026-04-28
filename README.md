# Assignment_03
This repository is WeiLin Liu's implementation of Assignment_03 of DIP.
<img width="974" height="822" alt="image" src="https://github.com/user-attachments/assets/6c20aeee-42f0-4474-ae57-13d170614f99" />

To install requirements:（After install CONDA（Anaconda or Miniconda））
```python
python conda install numpy matplotlib
python conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install pytorch3d
```
# Running
To complete task1:Budle Adjustment with Pytorch3D , run
```python
python
task1.py
```

To complete task2:Budle Adjustment with Colmap , run
```python
python
task2.py
```
# Results
## Task1:Budle Adjustment with Pytorch3D
The following pictures show the final effect and the loss curve.
If needed, download reconstructed_points_colored.obj to have detailed imformation.
<img width="717" height="693" alt="image" src="https://github.com/user-attachments/assets/6ff89b3c-fbc7-4f35-9b35-151ddb3934f8" />
<img width="731" height="668" alt="image" src="https://github.com/user-attachments/assets/558d181f-373f-435d-b4a2-543fcc82dfae" />
<img width="906" height="611" alt="image" src="https://github.com/user-attachments/assets/4e630a47-701c-4c88-8420-0e68d9955d84" />

As can be seen, because the default point size is small, the reconstructed model is rather sparse. Increasing the point cloud size makes the model more complete, but it also shows that the details are not uniform enough, with areas such as the eyes exhibiting a strong pixelated appearance.

## Task2:Budle Adjustment with Colmap
First, download colmap and add it to  PATH.
<img width="409" height="305" alt="image" src="https://github.com/user-attachments/assets/18717e99-1ed3-4783-8934-24aa931c415c" />
The following pictures show the final effect of Sparse Reconstruction and Dense Reconstruction.
If needed, download fused.ply and sparse.ply to have detailed imformation.

Sparse Reconstruction：
<img width="699" height="658" alt="image" src="https://github.com/user-attachments/assets/31ac50a6-73ec-415f-9f56-04d42387130f" />
<img width="641" height="618" alt="image" src="https://github.com/user-attachments/assets/558695bb-a377-42a8-9a87-1105d6bbbf2c" />

As can be seen, sparse reconstruction is indeed very "sparse". Even if the point cloud size is increased, only the shape of the model can be distinguished, and its internal details are almost not reflected.

Dense Reconstruction：
<img width="625" height="699" alt="image" src="https://github.com/user-attachments/assets/0f2a1029-e313-4b98-9b02-d03764aef293" />
<img width="656" height="696" alt="image" src="https://github.com/user-attachments/assets/c904431e-51ea-4e21-a3c7-89b10ff39d14" />


In comparison, dense reconstruction produces much better results, even with the default small point cloud size, it can reproduce the model's details quite well. After adjusting the point cloud volume to an appropriate value, it shows even better results than using native PyTorch3D. However, it also has the defect of rendering many white blocks at the boundaries, and it also shows uneven rendering in large areas of the same color (the model's chest).

# Acknowledgment
Thanks for Prof.Guo for providing the relevant formulas for coordinate systems and the ideas for initializing focal length and rotation matrices in the hints.🌹🌹🌹




