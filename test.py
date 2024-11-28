from train import XORMODEL_NEURAL_NETWORK
from utils import *
from constant import *


def test_data(inputs,output_expected):
    """
    Test the trained model and print the test outout and final accuracy after test.
    
    Args:
        inputs : Test input data.
        output_expected:expected output for test data.
    
    
    """
    model = XORMODEL_NEURAL_NETWORK()
    
    test_inputs = inputs[:10]
    predicted_test_output, _, _ = model.forward_propagation(test_inputs)
    accuracy_calculate = accuracy(output_expected[:10], predicted_test_output)
    print(f"Test Outputs:\n{predicted_test_output}, Test_accuracy:{accuracy_calculate}")
    
    