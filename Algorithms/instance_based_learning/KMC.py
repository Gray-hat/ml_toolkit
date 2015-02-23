#a program to implement the simplest form of the
#K-Means-Clustering

#standard library imports
import os
import random

#external library imports
from scipy.spatial.distance import euclidean

#package specific imports
from extras.utils.helpers import read_csv

def get_eucledian_distances(training_set, centroids):
	
	'''
		Compute the euclidean distance between the centroids and the
		training set points 
	'''	
	euclidean_distances = []
	for centroid in centroids:
		temp = []
		for row in training_set:
			row = row[1:]
			row  = list(map(lambda y: int(y), row))
			temp.append(euclidean(row, centroid))

		euclidean_distances.append(temp)		
	return euclidean_distances

def create_group_matrix(euclidean_distances):
	'''
	Given the matrix of euclidean distances
	the function returns the group matrix
	'''
	group_matrix = []
	#get the number of columns
	columns = len(euclidean_distances[0])
	for x in range(columns):
		#initialize smallest value to be
		
		minimun = euclidean_distances[0][x]
		index = 0
		for y in range(1,len(euclidean_distances)):
			if minimun > euclidean_distances[y][x]:
				index = y
				minimun = euclidean_distances[y][x]

		temp = [0] * len(euclidean_distances)
		temp[index] = 1
		group_matrix.append(temp)

	return group_matrix			



def get_new_points(training_set, group_matrix_new, clusters):
	'''
		Given the training set and group matrix return points belonging 
		to a particular centroid
	'''
	
	#initialize empty lists to store the points
	#the list is the size of the clusters
	points = [[] for i in range(clusters)]
	#loop through all the point whilst determining where to put them
	#in the points list
	#Each index in the points list corresponds to points close to the centroid
	for i in range(len(training_set)):
		index = group_matrix_new[i].index(1) 
		points[index].append(training_set[i])

	return points 	



def get_new_centroids(points):
	''' 
	given points belonging to a particular centroid computed from 
	the previously generated group matrix, calculate the new points for the centroids
	'''

	centroids = []
	for point in points:
		#no need to get averages
		if len(point) == 1:
			
			point = point[0][1:]
			#get the average
			point = list(map(lambda y: int(y), point))

			centroids.append(point)
		else:
			#list to store the points to be averaged

			columns = len(training_set[0]) - 1
			temp = [[] for i in range(columns)]
			for p in point:
				p = p[1:]
				for i in range(len(p)):
					temp[i].append(int(p[i]))

			#get averages
			average_list = []
			for element in temp:
				average = sum(element) / len(element)
				average_list.append(average)

			centroids.append(average_list)	

	return centroids	

	


if __name__ == '__main__':

	#location containing the training set
	location = os.path.dirname(os.path.abspath(__file__))
	training_set_location = os.path.join(location, 'extras/files/training_set2.csv')

	#read training data
	training_set = read_csv(training_set_location)

	#get the labels
	labels = training_set.pop(0)

	#prompt user to key in the required
	#number of clusters
	clusters = int(input('Key in the required number of clusters.\n'))

	#select the starting points at random to serve as cluster starting points
	random_points  = random.sample(training_set, clusters)
	centroids = []
	#remove the first element from the centrois correspoindint to the object name
	#convert strings to int
	for centroid in random_points:
		centroid = centroid[1:]
		centroids.append(list(map(lambda y: int(y),centroid)))

	print('The first centroids picked at random are {0}'.format(centroids))

	#initialize group_matrix old for the first loop
	group_matrix_old = []

	#loop until group matrix is stable
	while True:
		#compute eculidian distances
		euclidean_distances = get_eucledian_distances(training_set, centroids)

		#create group matrix

		group_matrix_new = create_group_matrix(euclidean_distances)
	
		#check stability
		flag = (group_matrix_old == group_matrix_new)
		group_matrix_old = group_matrix_new


		if not flag:
			#get new centroids
			points = get_new_points(training_set, group_matrix_new, clusters)
			centroids = get_new_centroids(points)

		else:
			print('The final centroids are {0}'.format(centroids))
			print('The points belonging to the centroids are {0}'.format(points))
			#exit loop
			break	
			

#use group matrix to print out the clusters

