# CIFAR-10 Image Classification Using Convolutional Neural Networks

## 1. Introduction

Image classification is an important application of Computer Vision and Deep Learning. The objective of image classification is to automatically identify the category or class to which an input image belongs.

In this project, Convolutional Neural Networks (CNNs) are developed using PyTorch to classify images from the CIFAR-10 dataset.

The project implements and compares two CNN architectures:

1. Baseline CNN
2. Improved CNN

The models are trained and evaluated using the CIFAR-10 dataset. Multiple experiments are performed with different numbers of training epochs to understand the effect of training duration on model performance.

The project also includes:

- Dataset loading and preprocessing
- CNN model development
- Model training
- Model evaluation
- Class-wise accuracy analysis
- Image prediction
- Multiple training experiments
- Result visualization
- Performance comparison
- Saving trained models for future use

---

# 2. Problem Statement

The objective of this project is to develop a deep learning-based image classification system capable of classifying CIFAR-10 images into one of ten predefined categories.

The project investigates how CNN architecture and training duration affect classification performance.

The main questions addressed are:

- How well can a basic CNN classify CIFAR-10 images?
- Does increasing the number of training epochs improve performance?
- Does an improved CNN architecture provide better performance than the baseline CNN?
- Which CIFAR-10 classes are easier or more difficult for the model to classify?
- Can the trained model correctly classify unseen test images?

---

# 3. Objectives

The main objectives of the project are:

1. To understand the CIFAR-10 image classification problem.
2. To load and preprocess the CIFAR-10 dataset using PyTorch.
3. To implement a baseline CNN architecture.
4. To implement an improved CNN architecture.
5. To train the models using different numbers of epochs.
6. To evaluate the models using test accuracy.
7. To calculate class-wise accuracy.
8. To perform predictions on sample test images.
9. To compare the performance of different experiments.
10. To generate graphs for visual analysis.
11. To save trained models for future predictions.
12. To analyze the overall results and identify the better-performing model.

---

# 4. Dataset Description

The CIFAR-10 dataset is used for this project.

CIFAR-10 contains 60,000 color images belonging to 10 different classes.

The dataset consists of:

| Property | Value |
|---|---:|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Image Size | 32 × 32 |
| Number of Channels | 3 |
| Number of Classes | 10 |

The images are RGB images with a spatial resolution of 32 × 32 pixels.

Therefore, each input image is represented as:

```text
3 × 32 × 32