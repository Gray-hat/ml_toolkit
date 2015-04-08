#A simple genetic algorithm to solve a 0-1 knapsack problem


#python standard imports
import collections
import random
from itertools import product
from operator import itemgetter
from random import shuffle
from random import randrange
from pylab import plot, ylim, show, legend, gca, figure


class KnapSack(object):
	'''Store attributes for the knapsack'''
	def __init__(self, cell, bag_capacity):
		'''Initialize'''

		self.bag_capacity = bag_capacity
		self.cell = cell

	def get_capacity(self, chromosome):
		'''Get the capacity occupied by the elements'''
		capacity = 0
		for i in range(len(chromosome)):
			gene = chromosome[i]
			if gene == "1":
				capacity = capacity + self.cell[i][1]
		return capacity		



	def get_benefit(self, chromosome):
		'''The benefit of the elements currently in the knapsack'''
		
		benefit = 0
		for i in range(len(chromosome)):
			gene = chromosome[i]
			if gene == "1":
				benefit = benefit + self.cell[i][0]
		return benefit		

class GeneticAlgorithm(object):
	'''Genetic algorithm to solve knapsack problem'''

	def __init__(self, cell, knapsack, generation_limit, population_size, crossover_ratio = 0.85, mutation_ratio = 0.1):
		'''initialize class attributes'''

		self.knapsack = knapsack
		self.generation_limit = generation_limit
		self.population = []
		self.population_size = population_size
		self.generate_initial_population()
		self.crossover_ratio = crossover_ratio
		self.mutation_ratio = mutation_ratio
		self.average_fitness = []

	def generate_initial_population(self):
		'''
			Generate the initial population. THis will be the chromosomes 
		'''
		population = product('01', repeat = len(self.knapsack.cell))
		for i in population:
			self.population.append(list(i))
		shuffle(self.population)

		#if the population exceeds the size then slice it down
		if len(self.population) > self.population_size:
			self.population = self.population[:self.population_size]
	def total_fitness(self):
		'''Return the summation of the fitness of each chromosome'''

		total = 0
		for i in self.population_fitness:
			total += i[1]

		return total	

	def fitness_function(self):
		'''The fitness of a function is calculated as the summation of the
		benifit of the items select to be put in the knapsack'''
		self.population_fitness = []
		for chromosome in self.population:
			capacity = self.knapsack.get_capacity(chromosome)
			#make any required adjustments to the capacity
			while capacity > self.knapsack.bag_capacity:
				random_pos = randrange(0, len(chromosome))
				if chromosome[random_pos] == '1':
					chromosome[random_pos] = '0'
				capacity = 	self.knapsack.get_capacity(chromosome)
			benefit = self.knapsack.get_benefit(chromosome)
			self.population_fitness.append([chromosome, benefit, capacity])
			
	def get_average_fitness(self):
		'''The average fitness on each iteration'''

		#flatten array
		temp = []
		for i in self.population_fitness:
			temp.append(float(i[1]))

		self.average_fitness.append(round(sum(temp)/ len(temp),3))

	def crossover(self,parents):
		'''Crossover parents in oder to obtain the children'''
		children = []
		parent_1 = parents[0]
		parent_2 = parents[1]
		locus = randrange(0, len(parent_1))
		child_1 = parent_1[:locus] + parent_2[locus:]
		child_2 = parent_2[:locus] + parent_1[locus:]
		children.append(child_1)
		children.append(child_2)
		return children




	def mutate(self,children):
		'''Mutate a gene on each of the children'''
		children_mut = []
		for child in children:
			random_pos = randrange(0, len(child))
			if random.random() <= self.mutation_ratio:
				if child[random_pos] == '1':
					child[random_pos] = '0'
				else:
					child[random_pos] = '1'
			children_mut.append(child)		
		return children_mut

	def reproduce(self, parents):
		'''Creates new offsprings'''
		#print("New children_p {0}".format(parents))

		if random.random() <= self.crossover_ratio:
			children = self.crossover(parents)
		else:
			children = list(parents)	
		#print("New children {0}".format(children))
		children = self.mutate(children)

		self.population.extend(children)
		
	def new_population(self):
		'''Population size should be fixed.
			fitness based elimination used
		'''			
		self.population_fitness.sort(key = itemgetter(1))
		#eliminate the least fit
		eliminate = self.population_fitness[0:2]
		for i in eliminate:
			self.population_fitness.remove(i)
			self.population.remove(i[0])

	def highest_frequency(self):
		'''Return the highest frequency'''
		#flatten array
		temp = []
		for i in self.population_fitness:
			temp.append(i[1])
		frequency = collections.Counter(temp)

		value = sorted(frequency.values())[-1]
		#percentage frequency of the fitness value most predominant
		percentage = value / len(temp)
		
		return percentage

	def selection(self):
		'''Roulete is used to determine the parents for the next generation'''
		parents = []
		copy_fitness = list(self.population_fitness)
		fit = 0
		for i in range(0,2):
			total_fit = self.total_fitness()
			total_fit -= fit
			random_fit = randrange(0, total_fit)
			copy_fitness.sort(key = itemgetter(1))
			for element in copy_fitness:
				random_fit -= element[1]
				if random_fit < 0:
					fit = element[1]
					parents.append(element[0])
					copy_fitness.remove(element)
					break

		return parents			

	def evolve(self):
		'''Evolve the population until the exit criteria is achieved'''
		gen_limit = 1
		self.fitness_function()
		self.get_average_fitness()
		flag = True
		while flag:
			if self.highest_frequency() > 0.9 and self.generation_limit < gen_limit:
				flag = False
			if self.highest_frequency() > 0.9:
				pass
			else:
				parents = self.selection()
				self.reproduce(parents)	
				self.fitness_function()
				self.get_average_fitness()
				self.new_population()
				

			gen_limit += 1

	


cell = [(4, 12),(2 ,1),(10, 4),(1 ,1), (2, 2)]
bag_limit = 15 #max bag can carry 15	

kp = KnapSack(cell, bag_limit)
ga = GeneticAlgorithm(cell, kp, 50, 40)
ga.evolve()
print(ga.population_fitness)
#plot graph
fig = figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Line graph to show average fitness')

ax.set_xlabel('Iterations')
ax.set_ylabel('Fitness')

ylim([0,bag_limit+10])
line1, = ax.plot(ga.average_fitness,label="Average Fitness", linestyle='--')
# Create a legend for the first line.
first_legend = legend(handles=[line1], loc=1)

# Add the legend manually to the current Axes.
ax = gca().add_artist(first_legend)
show()




