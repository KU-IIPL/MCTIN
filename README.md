# Meta-learning Convolution Transformer Integration Network for No-Reference Image Quality Assessment
Submited to IEEE Transactions on Image Processing  
*Donghyeon Lim, Changhoon Yim, IEEE Senior member, and Alan C. Bovik, IEEE Fellow*
## Abstract

The goal of no-reference image quality assessment
(NR-IQA) is to measure the image quality in accordance
with perception in the absence of a reference image. In this
paper, we propose an efficient NR-IQA method, called the
Meta-learning Convolution Transformer Integration Network
(MCTINet). MCTINet integrates a convolution-based ResNet and
a transformer-based Pyramid Vision Transformer (PVT) that
conducts NR-IQA. We train these two backbone networks and
integrate the two sets of output features on the support set and
query set in the meta-learning training process to construct
the meta-knowledge model. Given the meta-knowledge model
about various distortions, fine-tuning is conducted on a target
dataset with unknown distorted images as transfer learning. By
integrating ResNet and PVT backbone networks within the metalearning framework, our proposed model is able to conduct rapid
and efficient learning across various types of unknown distortions. Extensive experiments are conducted on multiple realworld authentic and synthetic distortion databases to compare
the performance of MCTINet with conventional and recent deep
learning-based NR-IQA methods. These include experiments on
individual databases, experiments on individual distortion types,
cross database evaluation, cross distortion type evaluation, and
ablation study. The experimental results show that MCTINet outperforms previous state-of-the-art methods for NR-IQA.

## Model Framework
![model_1](https://github.com/user-attachments/assets/efdd511f-7eac-4f2b-9f75-9f916a4cfe6e)

## Repository

The code for MCTINet is available at:

[https://github.com/KU-IIPL/MCTIN](https://github.com/KU-IIPL/MCTIN)

## Pre-trained Model
The PVT pre-trained model can be downloaded from [PVT](https://github.com/whai362/PVT) paper

The pre-trained model used in our experiments (meta_Train.pt) can be downloaded from the following link:

[https://drive.google.com/file/d/1nGLYTqg9aPsrgal3rQQ8Zb8nvJtCigHb/view?usp=sharing](https://drive.google.com/file/d/1nGLYTqg9aPsrgal3rQQ8Zb8nvJtCigHb/view?usp=sharing)

---


The code format adheres to the [METAIQA](https://github.com/zhuhancheng/MetaIQA) style.

