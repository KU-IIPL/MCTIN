# Meta-learning Convolution Transformer Integration Network for No-Reference Image Quality Assessment

## Abstract

The goal of no-reference image quality assessment (NR-IQA) is to evaluate image quality based on human perception without the need for a reference image. In this paper, we propose an efficient NR-IQA method called the **Meta-learning Convolution Transformer Integration Network (MCTINet)**. MCTINet integrates a convolution-based **ResNet** and a transformer-based **Pyramid Vision Transformer (PVT)** to perform NR-IQA.

We train these two backbone networks and integrate their output features from both the support set and query set during the meta-learning training process to construct a meta-knowledge model. Leveraging this model for various distortions, fine-tuning is conducted on a target dataset with unknown distorted images using transfer learning.

By combining ResNet and PVT within the meta-learning framework, MCTINet achieves rapid and efficient adaptation across different types of unknown distortions. Extensive experiments on multiple real-world authentic and synthetic distortion databases—including experiments on individual databases, specific distortion types, cross-database evaluations, cross-distortion evaluations, and ablation studies—demonstrate that MCTINet outperforms previous state-of-the-art methods in NR-IQA.

## model Framework
![model_1](https://github.com/user-attachments/assets/efdd511f-7eac-4f2b-9f75-9f916a4cfe6e)

## Repository

The code for MCTINet is available at:

[https://github.com/KU-IIPL/MCTIN](https://github.com/KU-IIPL/MCTIN)

## Pre-trained Model

The pre-trained model used in our experiments (metalearning.pt) can be downloaded from the following link:

[https://drive.google.com/file/d/1nGLYTqg9aPsrgal3rQQ8Zb8nvJtCigHb/view?usp=sharing](https://drive.google.com/file/d/1nGLYTqg9aPsrgal3rQQ8Zb8nvJtCigHb/view?usp=sharing)

---


The code format adheres to the [METAIQA](https://github.com/zhuhancheng/MetaIQA) style.

