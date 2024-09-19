# MNIST Digit Classification with TensorFlow

This project demonstrates how to build and train a neural network using TensorFlow to classify handwritten digits from the MNIST dataset. It covers key data science concepts like data preprocessing, model building, training, and evaluation.

![MNIST Digits Sample]

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- (Christos Cholevas)

## Overview
The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9). The goal of the project is to train a neural network to recognize and predict the digits accurately.

## Data
The MNIST dataset contains:
- 60,000 training images
- 10,000 test images

Each image is 28x28 pixels and is flattened into a 784-dimensional vector.

## Model Architecture
The neural network architecture includes:
- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer**: 512 neurons with sigmoid activation
- **Output Layer**: 10 neurons (one for each class)

```python
# Simple Network Architecture in TensorFlow
def neural_net(inputData):
    hidden_layer = tf.add(tf.matmul(inputData, weights['h']), biases['b'])
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['out']) + biases['out']
    return tf.nn.softmax(out_layer)

