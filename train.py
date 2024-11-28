import numpy as np
from utils import *
from constant import *

class XORMODEL_NEURAL_NETWORK:
    def __init__(self):
        '''  
           Random Initialization  
        Xavier Initialization for weights(In case of SIGMOID)
           hidden_layer_1
           hidden_layer_2
           output_layer
        '''
        
        self.weight_input_hidden1=np.random.randn(input_layer_neuron,hidden_layer1_neuron)*np.sqrt(1./input_layer_neuron)
        self.bias_hidden1=np.zeros((1,hidden_layer1_neuron))
    
        self.weight_hidden1_hidden2=np.random.randn(hidden_layer1_neuron,hidden_layer2_neuron)*np.sqrt(1./hidden_layer1_neuron)
        self.bias_hidden2=np.zeros((1,hidden_layer2_neuron))
    
        self.weight_hidden2_output=np.random.randn(hidden_layer2_neuron,output_layer_neuron)*np.sqrt(1./hidden_layer2_neuron)
        self.bias_output=np.zeros((1,output_layer_neuron))
        
        
    #now apply forward_propagation
    def forward_propagation(self,inputs):
        '''
            FORWARD PROPAGATION
        Perform forward propagation through the network.
        args:
            inputs: numpy array of the values
        return:
            tuple:predicted output and activation function of the hidden layers
        '''
         #hidden_layer_1
        hidden1_input=np.dot(inputs,self.weight_input_hidden1)+self.bias_hidden1
        hidden1_output=sigmoid(hidden1_input)
        #hidden_layer_2
        hidden2_input=np.dot(hidden1_output,self.weight_hidden1_hidden2)+self.bias_hidden2
        hidden2_output=sigmoid(hidden2_input)
        #Output_layer
        output_input=np.dot(hidden2_output,self.weight_hidden2_output)+self.bias_output
        predicted_output=sigmoid(output_input)
    
        return predicted_output, hidden1_output, hidden2_output
    
    def backward_propagation(self,inputs,output_expected,predicted_output,hidden1_output,hidden2_output,learning_rate):
        ''' 
            BACKWARD PROPAGATION
        perform the backward propagation that can update weight and minimize the loss
        Args:
           inputs: numpy array of input data
           output_expected: True values
           predicted_output: predicted values
           hidden1_output: activation of hidden layer1
           hidden2_output: activation of hidden layer2
           learning_rate: set for weight updates
            
        '''
        global weight_input_hidden1,bias_hidden1
        global weight_hidden1_hidden2,bias_hidden2
        global weight_hidden2_output,bias_output
    
        error=predicted_output-output_expected
        #now calculate the gradient descent of output and hiddern layers
        output_gradient=error*derivative_sigmoid(predicted_output)
        hidden2_gradient=np.dot(output_gradient,self.weight_hidden2_output.T)*derivative_sigmoid(hidden2_output)
        hidden1_gradient=np.dot(hidden2_gradient,self.weight_hidden1_hidden2.T)*derivative_sigmoid(hidden1_output)
    
        #now update wight and bias
        #output layer weight and bias is updated
        self.weight_hidden2_output-=learning_rate*np.dot(hidden2_output.T, output_gradient)
        self.bias_output-=learning_rate*np.sum(output_gradient, axis=0, keepdims=True)
    
        #update weight of hidden layer 2
        self.weight_hidden1_hidden2-=learning_rate*np.dot(hidden1_output.T,hidden2_gradient)
        self.bias_hidden2-=learning_rate*np.sum(hidden2_gradient, axis=0, keepdims=True)
    
        #update weight of hidden layer 3
        self.weight_input_hidden1-=learning_rate*np.dot(inputs.T,hidden1_gradient)
        self.bias_hidden1-=learning_rate*np.sum(hidden1_gradient, axis=0, keepdims=True)
        
    def train_data(self,inputs,output_expected,epochs,learning_rate):
        '''
            TRAIN THE WHOLE DATASET
        Train the neural network on the provided dataset
        This method can perform training by specific number of iteration in the defined epochs.First, it can apply 
        forward propagation and gave a predicted output ater that it can compute the loss and then apply backprop 
        that can update weight and minimize the loss. Also compute accuracy.
        args:
            output_expected: true_values
            epochs: No. of iteration the model can be train
            learning_rate: set for weight updates
        '''
        for i in range(epochs):
            #for forward_propagation
            predicted_output,hidden1_output,hidden2_output=self.forward_propagation(inputs)
            #for loss
            loss=binary_cross_entropy_loss(output_expected,predicted_output)
            #now for backward_propagation
            self.backward_propagation(inputs,output_expected,predicted_output,hidden1_output,hidden2_output,learning_rate)
            if i % 500 == 0:
                calculate_accuracy=accuracy(output_expected,predicted_output)
                print(f"Epoch {i}, Loss: {loss}, Accuracy: {calculate_accuracy}")
                


    
    
    
    
    