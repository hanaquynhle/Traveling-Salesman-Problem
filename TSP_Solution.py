"""
@author: Quynh Le (Hana)
@date: Mar 10, 2023
"""
import sys
import random as r
import numpy as np
import shutil

print("THE TRAVELING SALESMAN PROBLEM")
print("Please enter a file name:")
filename = input()
try:
  with open(filename, 'r') as f:
    data = f.readlines()
except:
  print("Can't find the file.")
  sys.exit()

#Remove newline characters from data
data = np.char.strip(data)

#Extract city mapping
mapping = {}
numOfCities = int(data[0])
for i in range(numOfCities):
  mapping[i] = data[i + 1].strip()

#Extract cost matrix
matrix = data[numOfCities + 1:]
cost = [[0 for i in range(numOfCities)] for j in range(numOfCities)]
for i in range(len(matrix)):
  rows = matrix[i].split(' ')
  for j in range(len(rows)):
    cost[i][j] = int(rows[j])


#Generate initial population of solutions
def population(numCities: int):
  pop1 = np.random.permutation(numCities)
  pop2 = np.random.permutation(pop1)
  result = [pop1.tolist(), pop2.tolist()]
  return result


#Calculate fitness
def fitness(pop):
  result = []
  for indi in pop:
    loss = 0
    for i in range(len(indi) - 1):
      loss += cost[indi[i]][indi[i + 1]]
    result.append(loss)
  return result


#Select individuals with lowest fitness (since the fitness is inversely proportional)
def select(pop):
  pop_fitness = fitness(pop)
  mapping_index = [(i, pop_fitness[i]) for i in range(len(pop_fitness))]
  mapping_index.sort(key=lambda x: x[1])
  select1 = pop[mapping_index[0][0]]
  select2 = pop[mapping_index[1][0]]
  return select1, select2


#Crossover
def crossover(select1, select2):
  randomIndex1 = r.randint(0, len(select1) - 1)
  randomIndex2 = r.randint(0, len(select1) - 1)
  while randomIndex2 == randomIndex1:
    randomIndex2 = r.randint(0, len(select1) - 1)
  index1 = min(randomIndex1, randomIndex2)
  index2 = max(randomIndex1, randomIndex2)

  subarray1 = select1[index1:index2 + 1]
  subarray2 = select2[index1:index2 + 1]

  mapping = [(subarray1[i], subarray2[i]) for i in range(len(subarray1))]
  mapping = np.random.permutation(mapping)

  subarray1 = [mapping[i][0] for i in range(len(mapping))]
  subarray2 = [mapping[i][1] for i in range(len(mapping))]

  select1 = select1[:index1] + subarray1 + select1[index2 + 1:]
  select2 = select2[:index1] + subarray2 + select2[index2 + 1:]

  return select1, select2


#Mutation
mutate_prob = 0.1


def mutate(pop):
  #determine if indi should be mutate based on probability
  for i in range(len(pop)):
    chance = r.random()
    if chance > mutate_prob:
      continue
    else:
      #Choose 2 random indices to swap
      size = len(pop[i])
      index1 = r.randint(0, size - 1)
      index2 = r.randint(0, size - 1)
      while index2 == index1:
        index2 = r.randint(0, size - 1)
      a = min(index1, index2)
      b = max(index1, index2)
      #Swap 2 indices
      pop[i] = pop[i][:a] + np.random.permutation(
        pop[i][a:b + 1]).tolist() + pop[i][b + 1:]

  return pop


def doTSP(numCities):
  pop = population(numCities)
  new_pop = []
  select1 = []  #List to store selected parents for crossover
  select2 = []
  cost1 = []  #Store fitness value (total cost) of the selected parents
  cost2 = []
  best_fitness = []  #Total cost of best solution so far
  num_gen = 50  #Number of generations in the algorithm

  while num_gen > 0:
    new_pop = []
    count = 0

    while len(new_pop) < len(pop) and count < 100:
      x, y = select(pop)
      new_x, new_y = crossover(x, y)

      og = fitness([x, y])
      new = fitness([new_x, new_y])

      if new[0] < og[0]:
        new_pop.append(new_x)
        new_pop.append(new_y)
      else:
        new_pop.append(x)
        new_pop.append(y)
      count += 1

    new_pop = mutate(new_pop)
    pop = pop + new_pop

    select1, select2 = select(pop)
    cost1, cost2 = fitness([select1, select2])
    num_gen -= 1
    circular_cost = cost[select1[-1]][select1[0]]
    best_fitness = cost1 + circular_cost

    print("Gen ", (50 - num_gen), " - Cost of shortest path so far:",
          best_fitness)

  return select1, best_fitness


def main():
  #Get the number of columns in terminal (formatting purpose)
  columns = shutil.get_terminal_size().columns

  #Run the entire program
  path, costpath = doTSP(numOfCities)

  for i in range(len(path)):
    path[i] = mapping[path[i]]
  path = ' '.join(path)
  path += ' ' 

  print("-----------------------------------".center(columns))
  print("Best path cost: ", costpath)
  print("Best path: ", path)


main()