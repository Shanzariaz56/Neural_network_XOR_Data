import pandas as pd
from train import XORMODEL_NEURAL_NETWORK
from test import *
from utils import *
from constant import *

if __name__ == "__main__":
    # Load XOR dataset
    data = pd.read_csv("Xor_Dataset.csv")
    print(data)

    # Data extraction
    inputs = data[["X", "Y"]].to_numpy()
    output_expected = data[["Z"]].to_numpy()

    # Initialize the neural network model
    model = XORMODEL_NEURAL_NETWORK()

    # Train the model
    print("Training the Model")
    train_response=model.train_data(inputs,output_expected, epochs, learning_rate)
    print(train_response)

    # Test the model
    print("Testing the Model\n")
    test_response=test_data(inputs,output_expected)
    print(test_response)
