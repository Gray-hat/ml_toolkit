#A file containing operations that assist in 
#machine learning operations

#standard library imports

import csv

def read_csv(location):
	'''
	A function that takes the path to a csv file and returns the
	contents of the file in a list.
	'''
	data_set = []
	with open(location, 'r') as csv_file:
		data = csv.reader(csv_file, delimiter = ',')
		for row in data:
			data_set.append(row)

	return data_set	
