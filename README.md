# CIFAR-10 Image Classification using PyTorch CNN

## 1. Project Overview

This project implements an image classification system using Convolutional Neural Networks (CNNs) with PyTorch.

The project uses the CIFAR-10 dataset, which contains 60,000 color images belonging to 10 different classes.

The main objectives are:

- Load and preprocess the CIFAR-10 dataset.
- Build a baseline CNN model.
- Build an improved CNN architecture.
- Train the models for different numbers of epochs.
- Evaluate model performance using test accuracy.
- Analyze class-wise classification performance.
- Perform predictions on sample images.
- Compare the performance of different experiments.
- Generate graphs for analysis and reporting.

---

## 2. Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- Image size: 32 × 32 pixels
- Number of channels: 3 (RGB)
- Number of classes: 10

### Classes

The ten CIFAR-10 classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Each image is represented as:

```text
3 × 32 × 32