import numpy as np 
import random 

class NeuralNetwork(): #creating a class neural network
    
    def __init__(self):
        # seeding for random number generation
        # set 0 for same random weights 1 for differnt random weights
        np.random.seed(0)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x+5)) #  default is -x but -x + 5 moving the sigmoid function to the right 

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for x  in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Random Generated Weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 3 input values and 1 output
    #can change the paramaetrs to logic gates 
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T # here when  first column of the input are high output is high

    #calling the  training function
    neural_network.train(training_inputs, training_outputs, 10000) #can vary the number of iterations

    print("Weights After Training: ")
    print(neural_network.synaptic_weights)

    #user inputs for new situation to give an output
    user_input_one = str(input(" Input One: "))
    user_input_two = str(input("Input Two: "))
    user_input_three = str(input("Input Three: "))
    
    print("New Values : ", user_input_one, user_input_two, user_input_three)
    print("Result: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    