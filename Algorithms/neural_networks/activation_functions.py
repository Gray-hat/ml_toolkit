#a file implemeting some of the known activation functions
#for the neural networks

def sigmoid_function(v):
		'''
		An activation function
		'''
		return (1 / (1 + math.exp(-v)))

def threshold_function(threshold, v):
	'''
	An activation function
	'''	
	#error checking not done
	if v >= threshold:
		return 1
	else:
		return 0

def line_equation(x):
	'''The equation of a line y = mx + c '''
	c,m = (0,1)

	return (m * x) + c 
