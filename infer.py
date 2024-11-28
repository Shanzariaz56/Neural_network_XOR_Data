from train import XORMODEL_NEURAL_NETWORK

def inference(inputs):
    """
    Make predictions using the trained mode
    Args:
        inputs: Input data to make predictions.
    return:
        predicted_output: Predicted_values
    """
    model=XORMODEL_NEURAL_NETWORK()

    predicted_output, _, _ = model.forward_propagation(inputs)
    return predicted_output
test=inference([1,1])
print(test)