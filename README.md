# Neural Network Training on XOR Dataset


## Overview
In this project simply implement the neural network through scratch to learn and in that case predict the XOR problem. The model is trained using a basic feedforward neural network, implementing forward propagation, backpropagation, and weight updates from scratch.

## Project Setup

### Requirements
- python 3.x
- numpy
- pandas
### install 
```bash
pip install numpy pandas
```
### Dataset
Use XOR dataset from kaggle which consist of binary input and output as truth table of XOR

## Code Description

### 1. load
the XOR dataset is loaded by using `pandas.read_csv()` 
```python
data=pd.read_csv("Xor_Dataset.csv")
print(data)
```
### 2. Neural Network Components
- Forward Propagation
- loss function (Binary-cross-entropy)
- Activation function (sigmoid)
- Backward Propagation

### 3. Training
The network is trained for a specified number of epochs. During each epoch, the forward pass, loss calculation, and backpropagation steps are executed. The weights are updated. 

### 4. Results:
The model's performance test throughout the training, with the loss and accuracy printed for every 500 epochs.


This README should help you get started with understanding and running the XOR neural network model.