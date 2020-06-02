import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import csv
import numpy as np
from itertools import permutations, cycle
import time
import random
import statistics
import matplotlib.pyplot as plot
import seaborn as sns
import time



file = open('european_cities.csv', 'r+')
reader = csv.reader(file,delimiter=";")

distance = np.zeros((24,24))
cities = []

i = 0
dicto = {}
distance1=[]
for row in reader:
    if i == 0:
        for j in range(len(row)):
            cities.append(row[j])
    else:
        for j in range(len(distance)):
            distance[i-1,j] = float(row[j])

    i += 1




def ex(N):

    #making a np array to use for permutations with index.
    sequence= np.arange(N)

    path = np.zeros(N)

    weight = 10000000000000
    optimal_path = np.zeros(len(cities))
    #checking solutions for every permutations and selecting the best
    for i in permutations(sequence):
        for d in range(len(i)-1):
            #value from city,
            #int(i[d]),int(i[d+1]) to get
            #the correct value and not the whole
            #array

            path[d] = distance[int(i[d]),int(i[d+1])]
            #value to get back to the starting city city
        path[-1] = distance[int(i[-1]), int(i[0])]


        if np.sum(path) < weight:
            weight = np.sum(path)
            optimal_path = i

    print("time for path:", weight)
    np.sort(optimal_path,axis=None)

    for x in range(len(optimal_path)):
        print(cities[int(x)])
        print("to")


start_time=time.time()
#ex(10)
print("--- %s seconds ---" % (time.time() - start_time))
# Barcelona to Belgrade to Berlin to Brussels to Bucharest to Budapest to Copenhagen to Dublin to Hamburg to Istanbul
# with a sum of 7486.309 with a time of --- 36.80531096458435 seconds ---
#probably 1.96612433 × 10to power of 16 years




def find_path(cities,Distances):
    tour = np.zeros(len(cities))
    for i in range(len(cities)-1):
        tour[i] = Distances[int(cities[i]),int(cities[i+1])]
    tour[-1] = Distances[int(cities[-1]), int(cities[0])]
    return np.sum(tour)



def switch(n1, n2, sequence):
    temp = sequence[n1]
    temp2 = sequence[n2]
    sequence[n1] = temp
    sequence[n2] = temp2
    #print(sequence)
    return sequence



def hill_climb(N, liste):


    tall = 20
    listee = []
    sequence = np.arange(N)
    sequence =np.random.choice(sequence, N, replace=False)
    path = np.zeros(N)
    weight = 10000000000000

    for i in range(len(sequence)-1):
        path[i] = distance[int(sequence[i]),int(sequence[i+1])]
    path[-1] = distance[int(sequence[-1]), int(sequence[0])]

    while tall > 0:

        for x in range(6):
            n1 = np.random.randint(N)
            n2 = np.random.randint(N)

            if n1 == n2:
                n1 = np.random.randint(N)

            ny_sequence =sequence.copy()

            switch(n1,n2,ny_sequence)

            ny_path = np.zeros(N)
            for i in range(len(ny_sequence)-1):

                ny_path[i] = distance[int(ny_sequence[i]),int(ny_sequence[i+1])]

            ny_path[-1] = distance[int(ny_sequence[-1]), int(ny_sequence[0])]

            if np.sum(ny_path) < np.sum(path):
                path = ny_path

        if np.sum(ny_path) < np.sum(path):
            path = ny_path
            sequence = ny_sequence
            tall = tall +1

        if weight > np.sum(path):
            weight = np.sum(path)

            liste.append(np.sum(path))
            listee.append(sequence)

        tall = tall-1

#uncomment to run hill_climb algo
"""
liste = []
for i in range(20):
    hill_climb(10,liste)

liste.sort()

print("avrage: ", (sum(liste) / len(liste)))
print(" Best", min(liste))
print("worst", max(liste))
print("standard devation", statistics.stdev(liste))

listee = []
for i in range(20):

    hill_climb(24,listee)

listee.sort()
print("for 24 cities ")
print("avrage: ", (sum(listee) / len(listee)))
print(" Best", min(listee))
print("worst", max(listee))
print("standard devation", statistics.stdev(listee))
"""

"""
output

avrage:  13014.7975
 Best 9620.43 vs exhaustive 7486.309
worst 16434.02
standard devation 1907.4620874120862

for 24 cities
avrage:  31462.107500000002
 Best 26759.48
worst 35473.38
standard devation 2448.3601335692665
"""




def path_distance(sequence, size):
    path = np.zeros(size)
    for i in range(len(sequence)-1):
        path[i] = distance[int(sequence[i]),int(sequence[i+1])]
    path[-1] = distance[int(sequence[-1]), int(sequence[0])]
    return path



def mutation(child, propabilaty):
    if(random.random() < propabilaty):

        size = np.random.randint(len(child))+1
        start = np.random.randint(len(child)-(size-1))
        end = start + size
        segment = child[start:end]
        reverse_segment = segment[::-1]
        child[start:end] = reverse_segment

    return child


def _population(population,  N, teller,population_min_distance,x):


    parents = np.zeros((5, 2))
    to_remove = np.zeros((5, 1))
    population_distances = np.zeros(len(population))

    # choosing parents using tournaments selection
    tournaments = np.zeros((5,int(len(population)/5)))
    population_i = np.arange(population.shape[0])
    np.random.shuffle(population_i)

    #creating tournament to pick parents
    count = 0
    for i in range(5):
        for j in range(int(len(population)/5)):
            tournaments[i,j] = population_i[count]
            count += 1


    for j in range(5):
        parents[j], to_remove[j] = create_parents(tournaments,population,j,distance)


    child_list = crossover1(parents,population)
    #mutating the children with a probability of 0.03
    for distances in range(len(child_list)):
        child_list[distances] =mutation(child_list[distances],0.03)


    #converting children list to numpy array
    npa = np.asarray(child_list, dtype=np.float32)


    new_population = np.zeros(population.shape)
    new_population = population.copy()

    y = 0
    to_remove = np.sort(to_remove, axis = 0)

    for i in range(len(to_remove)):
        #deleting  distances in population
        new_population = np.delete(new_population, to_remove[i]-y, 0)
        y += 1

    #adding child to new population

    for i in range(len(child_list)):
        new_population = np.vstack((new_population,child_list[i]))

    if teller > 0:

        teller = teller-1
        for k in range(new_population.shape[0]):
            population_distances[k] = find_path(new_population[k,:],distance)
        #selecting the best result
        population_min_distance[0,x] = np.amin(population_distances)

        x = x+1
        #recursivle running multible genorations
        _population(new_population, N, teller,population_min_distance, x)

    return population_min_distance







def create_parents(tournaments,population,tournament,Distances):


    # create the vectors with city indices of the whole tour
    result = np.zeros((tournaments.shape[0],population.shape[1]))
    distances = np.zeros(tournaments.shape[0])

    for i in range(tournaments.shape[0]):
        result[i] = population[int(tournaments[i,int(tournament)])]
        # Find the length of the actual tour
        distances[i] = find_path(result[i],Distances)


    # Find the parents
    parents = np.argpartition(distances, 2)[:2]

    Parents = np.zeros(len(parents))
    for i in range(len(Parents)):
        Parents[i] = tournaments[parents[i],int(tournament)]   #flipped
    removeable = np.argpartition(distances,-2)[-1]
    to_be_Removed = tournaments[removeable,int(tournament)]




    return Parents, to_be_Removed

# ordered crossover.
def crossover1(parents,pp):
    child_list = []
    rows = parents.shape[0]
    cols = parents.shape[1]
    for x in range(0,rows):
        for y in range(0,(cols-1)):

            child,child_start,child_end = [],[],[]
            #selecting a random gene along the length  and swapping
            #just finding a random index to get a sub list of both parents.
            startGene = min(int(random.random() * len(pp[int(parents[x][y])]) ),
            int(random.random() * len(pp[int(parents[x][y+1])])))

            endGene = max(int(random.random() * len(pp[int(parents[x][y])])),
            int(random.random() * len(pp[int(parents[x][y+1])])) )
            index_p1 = pp[int(parents[x][y])]
            list(index_p1)

            for i in range(startGene, endGene):
                child_start.append(index_p1[i])

            index_p2 = pp[int(parents[x][y+1])]
            list(index_p2)

            child_end = [item for item in index_p2 if item not in child_start]
            child = child_start + child_end

            child_list.append(child)
    return  child_list


#change number of cities here
Number_of_cities = 10
Number_of_population = [100,200,250]
Number_of_genorations = [100,200,300]

liste = []
liste2 =[]
runs = 0
#trying 100,200 and 300 genorations, and 100, 200 and 250 population

while runs !=3:
    start_time = time.time()

    populationn = np.zeros((Number_of_population[runs],Number_of_cities))
    vec = np.arange(Number_of_cities)

    population_min_distance = np.zeros((2,Number_of_genorations[runs]))

    #create a population
    for i in range(Number_of_population[runs]):
        np.random.shuffle(vec)
        populationn[i] = vec

    x = 0


    _population(populationn,Number_of_cities,Number_of_genorations[runs],population_min_distance,x)

    liste.append(population_min_distance)
    print("Number of population ", Number_of_population[runs])
    print ("best  ", min(liste[runs][0]))
    print ("worst ", max(liste[runs][0]))
    print ("mean is ", sum(liste[runs][0]) / len(liste[runs]))
    print("standard deviaton ", statistics.stdev(liste[runs][0]))
    runs = runs+1
    end = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))



one = liste[0]
two = liste[1]
three = liste[2]


plot.plot(one[0],label=1)
plot.plot(two[0],label=2)
plot.plot(three[0],label=3)
plot.legend(loc="best")
plot.xlabel("Generations")
plot.ylabel("distance")
plot.show()

#########
#Among the first 10 cities, did your GA find the shortest tour (as found by the exhaustive search)? Did it come close?

# it came close, 7486.
#For both 10 and 24 cities: How did the running time of your GA compare to that of the exhaustive search?
#the running time is much much shorter

##################


""" 10 cities
Terminal output

Number of population  100
best   7830.01
worst  10118.79
mean is  423785.8949999998
standard deviaton  587.5843827546489
--- 0.22074484825134277 seconds ---

Number of population  200
best   7486.3099999999995
worst  8665.22
mean is  775809.2900000033
standard deviaton  359.8096244485675
--- 0.7931280136108398 seconds ---

Number of population  250
best   7680.3
worst  9194.84
mean is  1209710.3599999973
standard deviaton  398.0234433581409
--- 1.424691915512085 seconds ---

for 24 european_cities

Number of population  100
best   20445.26
worst  26249.14
mean is  1148142.6649999996
standard deviaton  1808.9792165392116
--- 0.4494791030883789 seconds ---

Number of population  200
best   17754.1
worst  25506.96
mean is  2142549.8000000054
standard deviaton  2254.078428472829
--- 1.2412989139556885 seconds ---

Number of population  250
best   18366.91
worst  25675.27
mean is  3086934.535000005
standard deviaton  1915.1108514772097
--- 2.1928021907806396 seconds ---
"""
