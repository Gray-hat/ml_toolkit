#A simple program to implement neural networks
#our network only has one output

#python standard libs
import os
import math
import random
import numpy as np

from activation_functions import threshold_function
from activation_functions import sigmoid_function
from activation_functions import line_equation
from extras.utils.helpers import read_csv


class NeuralNetwork():
	'''
		An algorithm to implement neural networks
		Backpropagation is used as a default on neural networks with hidden layers 
		otherwise the perceptron or adaline is chosen.
	'''
	def __init__(self, data_set, algorithm, tolerance, learning_rate, bias = 1, threshold = 0.5, hidden_layer = False):
		#initialize class values
		self.hidden_layer = hidden_layer #A number showing number of nodes on the hidden layer
		self.tolerance = tolerance
		self.data_set = data_set
		self.normalized = []
		self.weights = []
		self.algorithm = algorithm #0 = Perptron 1 = adaline 2 = backprop
		self.learning_rate = learning_rate
		self.threshold = threshold
		self.bias = bias
		self.add_bias()
		self.set_initial_weights()
		self.normalize()


	def set_initial_weights(self):
		'''
		Set the initial weights for the neural networks nodes
		'''
		input_size = len(self.data_set[0]) - 1

		if self.hidden_layer and isinstance(self.hidden_layer, int):
			#create a mapping for node input to respective outputs
			weights = []
			#input to hidden layer
			for i in range(input_size):
				temp = [round(random.random(),3) for j in range(self.hidden_layer)]
				weights.append(temp)
			self.weights.append(weights)	
			#hidden layer to output
			temp = [round(random.random(),3) for i in range(self.hidden_layer)]
			self.weights.append(temp)


		else:
			#only the input and output layer exist
			weights = [round(random.random(),3) for i in range(input_size)]
			self.weights.append(weights)
			print("The initial weights are:")
			self.output_weights

	def output_weights(self):
		'''
			Print out the weights in a pretty format
		'''
		for i in range(len(self.weights)):
			if isinstance(self.weights[i][0], list):
				#weights is an array
				for j in range(len(self.weights[i])):
					print("Layer {0}  node {1} are: {2}".format(i, j, "".join(str(self.weights[i]))))
			else: 
				print("Layer {0} are: {1}".format(i, "".join(str(self.weights[i]))))		


	def add_bias(self):
		'''
			Add bias to the data
		'''
		for data in self.data_set:
			data.insert(0, str(self.bias)) #making it a string to enable comparisons on the normalization phase





	def normalize(self):
		'''Function to normalize the data'''
		temp = []
		#flatten array
		list(map(lambda x: temp.extend(x),self.data_set))
		largest = float(max(temp))
		self.normalized = list(self.data_set)

		#overide normalize list
		for i in range(len(self.normalized)):
			for j in range(len(self.normalized[i])):
				self.normalized[i][j] = (float(self.normalized[i][j])) / largest


	def iterate(self):
		'''
			Propagate phase
		'''

		if self.algorithm == 0 or self.algorithm ==1:
			x = 0
			flag = True
			#infinite loop
			while flag:
				#repeat the epoch until there is a convergence
				print('******************** Start of Epoch {0} ********************'.format(x))
				errors = []
				for data in self.normalized:
					#loop thru each data input
					inputs = data[:-1]
					d_output = data[-1:][0]
					print("The inputs to the network are {0} and the desired output is {1}".format(inputs, d_output))

					temp = np.matrix(inputs)
					weights = np.matrix(self.weights[0]).T
					product = np.dot(temp, weights)
					summation = sum(product)

					if self.algorithm ==0:
						activated = threshold_function(self.threshold, summation.item())
					else:
						activated = line_equation(summation.item())	
				
					print("The current weights and model outputs are {0} and {1} respectively".format(weights, activated))
					
					error = d_output - activated
					errors.append(error * error)
					weight_change = temp.T * (self.learning_rate * error) 
					print("The models error is {0} and the change in weights is {1}".format(error, weight_change))
					self.weights = np.matrix(weight_change) + np.matrix(self.weights).T
					self.weights = self.weights.T
					print("The updated weights are:")
					self.output_weights()
				print('******************** End of Epoch {0}   ********************'.format(x))
	
				average_error = np.mean(errors)
				print("The average error is, {0}".format(average_error))
				if round(average_error,1) <= self.tolerance:
					print("The desired error tolerance has been achieved.")
					print("Exiting... ")
					print("The weights for the model are:")
					self.output_weights()
					flag = False
					break
						
				
				x += 1	
		elif self.algorithm == 2:
			self.backprop()

		print("Testing completed successfully...")	
		print("The acceptable weights for the model are:")
		self.output_weights()	


	def perceptron(self):
		'''
			Method to implement the perceptron 
		'''
		
	def adaline(self):
		'''
		'''
		pass

	def backprop(self):
		'''
		'''
		pass				




if __name__ == '__main__':

	#location containing the training set
	location = os.path.dirname(os.path.abspath(__file__))
	training_set_location = os.path.join(location, 'extras/files/training_set1.csv')

	#read training data
	training_set = read_csv(training_set_location)

	#remove the data lables
	labels = training_set.pop(0)

	neuron = NeuralNetwork(training_set, 0, 0.1, 0.1)
	neuron.output_weights()
	print(neuron.normalized)
	neuron.iterate()

	   
	
		
