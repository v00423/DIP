# Assignment_02
This repository is WeiLin Liu's implementation of Assignment_02 of DIP.
![poisson_3](https://github.com/user-attachments/assets/47fe9bf0-f737-4b69-b425-8250a4056d8a)

To install requirements:
```python
python -m pip install -r requirements.txt
```
# Running
To complete Poisson Image Editing, run
```python
python
run_blending_gradio.py
```
To complete Pix2Pix Image Editing, run
```python
python
dowload_dataset.py
train.py
```
# Results
## Poisson Image Editing
This video shows how to edit the image.


【DIP第二次作业演示】 https://www.bilibili.com/video/BV1iuQGBxEio/?share_source=copy_web&vd_source=10e42189134f53514fe21036635ddf61


The following pictures are the results.
![poisson_1](https://github.com/user-attachments/assets/a9d2d163-36fc-4a14-8ec3-9af7ecac85a7)

![poisson_3](https://github.com/user-attachments/assets/7ca35352-df11-42d5-8d95-88298513d534)

# Pix2Pix Image Editing
## Facades dataset

For the facades dataset,the final laplacian loss converges and fluctuates around 0.36.

Final epoch of the traning set image:
<img width="768" height="256" alt="result_1" src="https://github.com/user-attachments/assets/f1e1d615-7603-48eb-a184-b3b765693dc4" />

<img width="768" height="256" alt="result_2" src="https://github.com/user-attachments/assets/890f1ad4-aee2-4d40-bdd0-b9be331b0e25" />

<img width="768" height="256" alt="result_3" src="https://github.com/user-attachments/assets/fa77ff90-d317-416c-94ac-ca829566f5ad" />

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/5434e61a-48b2-4d31-b4dd-b4e7d8a0a97b" />

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/611eddd8-b560-4d18-ac6e-9b0930945ceb" />

Final epoch of the validation set image:
<img width="768" height="256" alt="result_1" src="https://github.com/user-attachments/assets/281a0eab-b3c6-4eb9-864c-51c44e19f387" />

<img width="768" height="256" alt="result_2" src="https://github.com/user-attachments/assets/187e5ff9-ce0d-44bd-8810-ec6df0bb2d3c" />

<img width="768" height="256" alt="result_3" src="https://github.com/user-attachments/assets/4053d549-12e3-4a71-aa11-9a59dae94f4e" />

<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/ffa0d66b-a871-4389-93c2-544077f2ec90" />

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/ce6dcd65-778f-4cc7-8aab-6fbe789eed33" />

The training set results are decent, with high matching accuracy, but not sharp enough; however, the validation set results are poor.

## Cityscapes dataset

For the cityscapes dataset,the final laplacian loss converges and fluctuates around 0.10.

Final epoch of the traning set image:
<img width="768" height="256" alt="result_1" src="https://github.com/user-attachments/assets/e22035f6-9bd2-472a-a4ca-2a544944e19e" />

<img width="768" height="256" alt="result_2" src="https://github.com/user-attachments/assets/4cbf43ab-64b6-4487-a8a5-e2f818ab9cf7" />

<img width="768" height="256" alt="result_3" src="https://github.com/user-attachments/assets/f5b36287-60f5-49fc-85f9-b08ebc6e8000" />

<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/b98b81fa-f814-4f66-86e2-a2f357572529" />

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/a02f4f3d-d6e4-4603-b407-d06335a49bc7" />

Final epoch of the validation set image:
<img width="768" height="256" alt="result_1" src="https://github.com/user-attachments/assets/b7558d44-415d-41ce-b2e0-64996ed9f182" />

<img width="768" height="256" alt="result_2" src="https://github.com/user-attachments/assets/d04ebcaa-4465-4659-a771-39c7fd1673c9" />

<img width="768" height="256" alt="result_3" src="https://github.com/user-attachments/assets/f2998102-ab7c-4baa-a5d7-fc2a037128f9" />

<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/b2489591-7e26-403b-b332-74e6c5340e44" />

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/34044c3a-5475-4f1c-ab1e-9972110d1119" />

Same as facades, the training set results are decent, with high matching accuracy, but not sharp enough; however, the validation set results are poor.

# Acknowledgment
Thanks for the Poisson algorithm and laplacian loss algorithm proposed by Prof.GUO's teaching Slides

Also thanks for Prof.GUO for sharing the parameter settings for FCN networks and  detailed answers.🌹🌹🌹




