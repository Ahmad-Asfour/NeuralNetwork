'''
Implementation of a 2 Layer Neural Network in Python with Sigmoid Activation and Sum-Of-Squares Loss
'''
import numpy as np
'''

'''
class NeuralNetwork:
    
    def __init__(self, x, y):
        '''
        Initializes the neural network with input data and target output.
        
        Parameters:
        x (numpy.ndarray): Input data of shape (number of samples, number of features).
        y (numpy.ndarray): Target output of shape (number of samples, 1).
        '''
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)
    

    def feedforward(self):
        '''
        Performs forward propagation through the network.
        Calculates the output of the hidden layer and the final output.
        '''
        z = self.input @ self.weights1
        self.layer1 = self.sigmoid(z)
        self.output = self.sigmoid(self.layer1 @ self.weights2)
    
    def backprop(self):
        '''
        Performs backpropagation to compute the gradient of the loss function
        with respect to the weights, and updates the weights.
        '''
        d_weights2 = self.layer1.T @ (2 * (self.y - self.output) * self.sigmoid_derivative(self.output))
        d_weights1 = self.input.T @ (((2 * (self.y - self.output) * self.sigmoid_derivative(self.output)) @ self.weights2.T) * self.sigmoid_derivative(self.layer1))
        
        # Update the weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def sum_of_squares(self):
        '''
        Calculates the sum-of-squares loss function, which measures the 
        squared differences between the predicted output and the actual target output.
        
        Returns:
        float: The sum of squared differences.
        '''
        return np.sum(np.square(self.output - self.y))
    
    def sigmoid(self, x):
        '''
        Applies the sigmoid activation function to each element in the input array.
        
        Parameters:
        x (numpy.ndarray): The input array.
        
        Returns:
        numpy.ndarray: The output after applying the sigmoid function.
        '''
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        '''
        Computes the derivative of the sigmoid function.
        
        Parameters:
        x (numpy.ndarray): The input array, which is assumed to be the output 
                           of the sigmoid function itself.
        
        Returns:
        numpy.ndarray: The derivative of the sigmoid function.
        '''
        return x * (1 - x)

    def train(self, epochs=1000):
        '''
        Trains the neural network by performing multiple iterations of forward
        propagation and backpropagation.
        
        Parameters:
        epochs (int): The number of times to repeat the training process.
        '''
        for _ in range(epochs):
            self.feedforward()
            self.backprop()
            if _ % 100 == 0:
                print(f'Epoch {_}: Loss = {self.sum_of_squares()}')

# Example usage
# Input data (4 samples, 3 features each)
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Target output (4 samples, 1 target value each)
y = np.array([[0], [1], [1], [0]])

# Initialize and train the neural network
nn = NeuralNetwork(x, y)
nn.train(epochs=1000)

# Display the final output
print("Output after training:")
print(nn.output)
