# MLP Model

## Introduction

This repository contains an implementation of a Multi-Layer Perceptron (MLP) model for image classification. The model is implemented in Python and utilizes the PyTorch library for building and training the neural network.

## Dataset

The CIFAR-2 dataset is used for training and testing the MLP model. The dataset consists of 10,000 32x32 color images in 2 classes, with 5,000 images per class. There are 8,000 training images and 2,000 test images.

## Model Architecture

The MLP model consists of two hidden layers with ReLU activation functions and a final output layer with a sigmoid activation function. The number of hidden units can be customized during the model initialization.

## Training

The model is trained using stochastic gradient descent with momentum and L2 regularization. The training loop runs for a specified number of epochs, and mini-batch gradient descent is used for efficiency.

## Evaluation

After training, the model's performance is evaluated on the test dataset. The evaluation includes calculating the accuracy and loss metrics to assess the model's performance.

## Usage

To use the MLP model, follow these steps:

1. Install the required dependencies using `pip install -r requirements.txt`.

2. Load the CIFAR-2 dataset using the provided `pickle` file.

3. Initialize the MLP model with the desired input dimensions and the number of hidden units.

4. Train the model using the `train()` function with appropriate hyperparameters.

5. Evaluate the model using the `evaluate()` function to calculate accuracy.

## Contributions

Contributions to the project are welcome! If you find any bugs or want to add new features, feel free to create a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
