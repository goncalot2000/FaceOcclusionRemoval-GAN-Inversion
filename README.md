# GAN Inversion for Occlusion Removal in Face Images

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-green.svg?style=plastic)
![CUDA 11.6](https://img.shields.io/badge/CUDA-11.6-green.svg?style=plastic)

![image](./teaser.jpg)

**Figure:** *GAN Inversion model architecture using 'clean' images as prior knowledge during optimization*

> **GAN Inversion for Occlusion Removal in Face Images** <br>
> Gonçalo Teixeira <br>
> Abstract: Facial occlusions present a challenging problem in computer vision, affecting various applications such as face recognition and image analysis. The proposed method incorporates prior knowledge by using a set of face images without any occlusion as a reference during the reconstruction process. By inverting the latent codes of the occlusion-free reference images, a representative latent space is established. Notably, this approach distinguishes itself from existing methods by incorporating additional information about the person's face through the use of reference images.

[[Paper](https://arxiv.org/pdf/2004.00049.pdf)]

**NOTE:** This repository used the work developed in [this repo](https://github.com/genforce/idinvert) as a baseline.

## Pre-trained Models

Please download the pre-trained models from the following links and save them to `models/pretrain/`

| Description | Generator | Encoder | Discriminator |
| :---------- | :-------- | :------ |    :------    |
| Model trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. | [face_256x256_generator](https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing)    | [face_256x256_encoder](https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing)    | [face_256x256_discriminator]((https://shi-labs.com/projects/stylenat/checkpoints/FFHQ256_940k_flip.pt))
| [Perceptual Model](https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)

As StyleGAN uses aligned face for GAN training, all faces used in this repo are pre-aligned. The alignment method can be found at [stylegan-encoder](https://github.com/Puzer/stylegan-encoder).

## Installation

1. Create a virtual envirnoment via 'conda' with python 3.10
2. Install 'cuda' 11.6
3. Install all the dependencies using the following code.
```bash
python -m pip install -r requirements.txt
```

## Data Input

All the synthetic face occlusions used were produced by [face-occlusion-generator](https://github.com/kennyvoo/face-occlusion-generation).

| Path | Description
| :--- | :----------
| dataset | Main directory
| &ensp;&ensp;&boxvr;&nbsp; Abdullah_Gul | First person from the dataset
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; occluded_img | Directory containing the images to be reconstucted 
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; occlusion_mask | Directory containing the binary occlusion masks for the images being reconstucted 
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; reference_imgs | Directory containing the reference images
| &ensp;&ensp;&boxvr;&nbsp; Al_Gore | Second person from the dataset
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; occluded_img | Directory containing the images to be reconstucted 
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; occlusion_mask | Directory containing the binary occlusion masks for the images being reconstucted 
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; reference_imgs | Directory containing the reference images

**NOTE:** Both occluded images and the corresponding occlusions masks need to have the same name.

## Face Oclusion Removel

```bash
MODEL_NAME='styleganinv_ffhq256'
DATASET='dataset/'
python invert.py $MODEL_NAME $DATASET
```