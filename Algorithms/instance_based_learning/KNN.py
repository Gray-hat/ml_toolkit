#a program to implement the simplest form of the
#K-Nearest-Learning algorithm

#Standard library imports
from operator import itemgetter
import os

#package specific imports
from extras.utils.helpers import read_csv

#external libraries imports
from scipy.spatial.distance import euclidean

def get_input(labels):
	'''
	prompts user to key in the required inputs
	'''
	query_set = []
	
	for label in labels[:-1]:
		temp = int(input("Enter value for feature {0}\n".format(label)))
		query_set.append(temp)
	neighbours = int(input('Enter the value of K (neighbours required)\n'))

	return query_set, neighbours



def get_euclidean_distances(training_set, query_set):
	
	'''
	A function that gets the euclidean distances from each instance
	of the training_set and the query set and returns an ordered
	nested list dependant on the distance calculated
	'''
	results = []
	for data in training_set:
		temp = []
		temp = data[:-1]
		temp = list(map(lambda y: int(y),temp))
		distance = euclidean(temp,query_set)
		results.append([distance,data[-1:][0]])
	results.sort(key = itemgetter(0))
	return results	



def vote(results, neigbours):	
	'''
	A function that takes its inputs as a nested list with the
	ordered euclidean distances and determines the wining label set.
	#tyranny of numbers
	'''

	to_consider = results[:neigbours] #since its an ordered list
	to_consider = [x[-1] for x in to_consider]
	voting_results = {data:to_consider.count(data) for data in to_consider}

	max_key = max(voting_results)

	return max_key

if __name__ == '__main__':
	file_loation = os.path.dirname(os.path.abspath(__file__))
	location = os.path.join(file_loation, 'extras/files/training_set1.csv')

	#get the training set
	training_set = read_csv(location)

	#remove the data lables
	labels = training_set.pop(0)

	#get the users input
	query_set, neigbours = get_input(labels)



	#calculate euclidean distances
	response = get_euclidean_distances(training_set, query_set)

	#voting
	winning_label = vote(response, neigbours)

	print("The query set belongs to the label '{0}'.".format(winning_label))

