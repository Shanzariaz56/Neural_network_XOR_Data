import numpy as np
def sigmoid(x):
    '''
        Sigmoid Activation function
    Args:
        x(parameter): consider as input value and array of input to which sigmoid function is applied
    Return:
        numpy array: sigmoid-transformed the output where each value of 
                            "x" is transform into 0 and 1
    '''
    return 1/(1+np.exp(-x))

# here we can take derivative of sigmoid function for backpropagation
def derivative_sigmoid(x):
    '''
        Derivative of sigmoid function 
    That can be used for backward propagation (weight update and gradient descent)
    Args:
        (parameter):input array of the sigmoid function
    return:
        numpy array: derivative of sigmoid is applied to the input
    '''
    return x*(1-x)

def binary_cross_entropy_loss(y_true, y_pred):
    '''
          LOSS FUNCTION (Binary-Cross-Entropy)
        Args:
            y_true: Expected_output
            y_pred: Predicted_output
        Now apply formula that can calculate the loss 
        return:
            that return the calculated loss 
    '''
    loss_calculate = -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
    return loss_calculate

 
def accuracy(y_true,y_predicted):
    '''
        ACCURACY CALCULATION
        args:
            y_true: Expected_output
            y_pred: Predicted_output
        1-first convert all the prediction into the binary as "0","1"
        2- Then calculate total prediction
        3-Then apply the formula to calculate the accuracy
        return:
            return the accuracy value
    ''' 
    prediction_binary=(y_predicted>=0.5).astype(int)
    total_prediction=len(prediction_binary)
    true_prediction=np.sum(y_true==prediction_binary)
    accuracy_formula=(true_prediction/total_prediction)*100
    return accuracy_formula





