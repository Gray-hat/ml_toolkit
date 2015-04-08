#a file implemeting some of the known activation functions
#for the neural networks
import math
import numpy as np
def sigmoid_function(v):
		'''
		An activation function
		'''
		if  isinstance(v,np.matrixlib.defmatrix.matrix):
			for x in np.nditer(v, op_flags = ['readwrite']):
				x[...] = round((1 / (1 + math.exp(-(x)))),2)
			return v
		return (1 / (1 + math.exp(-(v))))

def threshold_function(threshold, v):
	'''
	An activation function
	'''	
	#error checking not done
	if  isinstance(v,np.matrixlib.defmatrix.matrix):
			for x in np.nditer(v, op_flags = ['readwrite']):
				if x >= threshold:
					x[...] = 2
				else:
					x[...] = 1
			return v
	if v >= threshold:
		return 2
	else:
		return 1

def line_equation(v):
	'''The equation of a line y = mx + c '''
	c,m = (0,1)

	if  isinstance(v,np.matrixlib.defmatrix.matrix):
			for x in np.nditer(v, op_flags = ['readwrite']):
				x[...] = (m * x) + c 
			return v
	return (m * v) + c 
