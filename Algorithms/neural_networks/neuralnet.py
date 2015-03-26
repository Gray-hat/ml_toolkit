#A simple program to implement neural networks
#our network only has one output

#python standard libs
import os
import math
import random
import numpy as np
from pylab import plot, ylim, show

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
		self.average_errors = []
		self.algorithm = algorithm #0 = Perptron 1 = adaline 2 = backprop
		self.learning_rate = learning_rate
		self.threshold = threshold
		self.test_set = []
		self.bias = bias
		self.add_bias()
		self.set_initial_weights()
		self.normalize()
		self.split_dataset()

		assert (algorithm == 2 and hidden_layer) or (algorithm ==1) or (algorithm ==0), "Hidden layer must be defined"

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
					print("Layer {0}  node {1} are: {2}".format(i, j, "".join(str(self.weights[i][j]))))
			else: 
				print("Layer {0} are: {1}".format(i, "".join(str(self.weights[i]))))		


	def add_bias(self):
		'''
			Add bias to the data
		'''
		for data in self.data_set:
			data.insert(0, self.bias) #making it a string to enable comparisons on the normalization phase


	def split_dataset(self):
		'''splits the dataset to test and training data'''

		size = int(0.2 * len(self.normalized)) or 1
		self.test_set.append(self.normalized[-size])

		self.normalized = self.normalized[:len(self.normalized) - size]


	def normalize(self):
		'''Function to normalize the data'''
		temp = []
		#flatten array
		list(map(lambda x: temp.extend(x),self.data_set))
		temp = [int(i) for i in temp]
		print(temp)
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


					if self.algorithm ==0:
						activated = threshold_function(self.threshold, product.item())
					else:
						activated = line_equation(product.item())	
				
					print("The current weights and model outputs are {0} and {1} respectively".format(weights, activated))
					
					error = d_output - activated
					self.average_errors.append(error)
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
					flag = False
					break
						
				
				x += 1	
		elif self.algorithm == 2:
			self.backprop()

		print("Testing completed successfully...")	
		print("The acceptable weights for the model are:")
		self.output_weights()	


	def backprop(self):
		'''
			Backprop learning and training
		'''

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

				input_0 = np.matrix(inputs)
				output_0 = line_equation(input_0)
				print("The outputs of the input layer are {0}".format(output_0))
				weights = np.matrix(self.weights[0]).T
				input_1 = np.dot(weights, input_0.T)
				print("The inputs of the hidden layer are {0}".format(input_1))
				output_1 = sigmoid_function(input_1)
				print("The outputs of the hidden layer are {0}.".format(output_1))
				input_2 = np.dot(self.weights[1], output_1)
				print("The input to the output layer is {0}".format(input_2))
				output_2 = sigmoid_function(input_2.item())
				print("The output to the output layer is {0}".format(output_2))

				error = d_output - output_2
				self.average_errors.append(error)
				errors.append(error * error)

				#calculate weight for the second weights

				d = (d_output - output_2) * output_2 * (1 - output_2)
				weight_change_w = self.learning_rate * d * output_1

				#calculate weights for the first weights

				w = np.matrix(self.weights[1]).T * d
				a, b, g, h = (w[0].item(), w[1].item(), output_1[0].item(), output_1[1].item())

				weight_change_v = [[a * g * (1 - g)],[b * h * (1 - h)]]
				weight_change_v = np.matrix(weight_change_v).T
				#update weights
				print("Updating weights.......")

				self.weights[0] = ((output_0.T * weight_change_v) * self.learning_rate) + self.weights[0]


				self.weights[1] = np.matrix(self.weights[1]).T + weight_change_w
				self.weights[1] = self.weights[1].T

				print("The updated weights are:")
				self.output_weights()
			print('******************** End of Epoch {0}   ********************'.format(x))

			average_error = np.mean(errors)
			print("The average error is, {0}".format(average_error))
			if round(average_error,1) <= self.tolerance:
				print("The desired error tolerance has been achieved.")
				print("Exiting... ")
				flag = False
				break
					
			
			x += 1	
						
	def test(self):
		'''Test the model to check if it conforms to
		the standards'''
		#true class
		class_p = 1
		class_n = 0

		tp = 0
		fn = 0
		fp = 0
		tn = 0

		print("******************TESTING**************")
	

		for data in self.test_set:
			#loop thru each data input
			inputs = data[:-1]
			d_output = data[-1:][0]
			print("The inputs to the network are {0} and the desired output is {1}".format(inputs, d_output))

			if self.algorithm == 0 or self.algorithm == 1:

				temp = np.matrix(inputs)
				weights = np.matrix(self.weights[0]).T
				product = np.dot(temp, weights)

				if self.algorithm ==0:
					m_output = threshold_function(self.threshold, product.item())
				else:
					m_output = line_equation(product.item())
				print("The models output is {0}".format(m_output))		

			else:
				input_0 = np.matrix(inputs)
				output_0 = line_equation(input_0)
				print("The outputs of the input layer are {0}".format(output_0))
				weights = np.matrix(self.weights[0]).T
				input_1 = np.dot(weights, input_0.T)
				print("The inputs of the hidden layer are {0}".format(input_1))
				output_1 = sigmoid_function(input_1)
				print("The outputs of the hidden layer are {0}.".format(output_1))
				input_2 = np.dot(self.weights[1], output_1)
				print("The input to the output layer is {0}".format(input_2))
				m_output = sigmoid_function(input_2.item())
				print("The output to the output layer is {0}".format(m_output))	

			if m_output == d_output == class_p:
				tp += 1
			elif m_output == d_output == class_n:
				tn += 1
			elif m_output == class_p and d_output == class_n:
				fp += 1
			elif m_output == class_n and d_output == class_p:
				fn += 1	
			else:
				print("An error has occured")			



		print("**********COMPLETE***************")
		print("Analysis....")

		recall = tp / (tp + fn)
		precision = tp / (tp + fp)
		f_score = (2 * (precision * recall)) / (precision + recall)
		print("The recall is: {0}".format(recall))
		print("The precision is: {0}".format(precision))
		print("The F-score is: {0}".format(f_score))



if __name__ == '__main__':

	#location containing the training set
	location = os.path.dirname(os.path.abspath(__file__))
	training_set_location = os.path.join(location, 'extras/files/training_set1.csv')

	#read training data
	training_set = read_csv(training_set_location)

	#remove the data lables
	labels = training_set.pop(0)
	
	neuron = NeuralNetwork(training_set, 1, 0.1, 0.2)
	print(neuron.weights)
	neuron.output_weights()
	print(neuron.normalized)
	neuron.iterate()
	#neuron.test()

#backprop

	# neuron = NeuralNetwork(training_set, 2, 0.1, 0.8, bias = 0, hidden_layer = 2)
	# print(neuron.weights)
	# neuron.output_weights()
	# print(neuron.normalized)
	# neuron.iterate()
	# #neuron.test()

#plot graph
ylim([-1, 1])
plot(neuron.average_errors)
show()

	
		
