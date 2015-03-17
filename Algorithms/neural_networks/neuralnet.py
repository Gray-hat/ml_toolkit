#A simple program to implement neural networks
#our network only has one output

#python standard libs
import random

class NeuralNetwork():
	'''
		An algorithm to implement neural networks
		Backpropagation is used as a default on neural networks with hidden layers 
		otherwise the perceptron or adaline is chosen.
	'''
	def __init__(self, data_set, hidden_layer = False):
		#initialize class values
		self.hidden_layer = hidden_layer #A number showing number of nodes on the hidden layer
		self.tolerace = 0
		self.data_set = data_set
		self.weights = []
		self.set_initial_weights()


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



if __name__ == '__main__':
	neuron = NeuralNetwork([[1,2,3,4]],5)
	print(neuron.weights)

	   
	
		
