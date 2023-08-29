from .classes import Values
from .evolution import _generate_parent,_generate_population,_generate_selection,_mutate,_generate_crossover,Chromosome
import random
import pandas as pd
import numpy as np

df = pd.read_csv('data/nutrients.csv')
food_dictionary = df.set_index('id').T.to_dict()

def null(x):
    return np.where(np.isnan(x),0.0,x)


def nutrition(meals,food_dictionary):
    p,s,c = [],[],[]
    for x in meals:
        b = Values(food_dictionary.get(x))
        sugars = null(b.sugars)
        protein = null(b.protein)
        cal = null(b.calories)
        # print (b.group)
        p.append(protein)
        s.append(sugars)
        c.append(cal)
    return np.array((p,c,s)).T
        

def get_fitness(x):
    # Initialize the dictionary
    m = nutrition(x,food_dictionary)
    # return (np.sum(p)*4.0-np.sum(s))/np.sum(c)
    sigma = np.sum(m,axis=0)
    # sugar_constraint = np.where(sigma[2]>0,1,0)
    # print (sugar_constraint)
    return sigma[0]*4.0/sigma[1]
    


def genetic_algorithm(generations,ids,food_dictionary,population_size,n_items):
    """
    initialize population 
    Params:
        generations: N generations to run (more is not always better)
        n_items: number of food items (genes)
        ids: unique identifier of each food item
        get_fitness: objective function to optimize against
        population_size: n population of starting generation
    """
    pop = _generate_population(n_items,ids,get_fitness,population_size)
    best_eval = pop[0].fitness
    metrics = []
    grocery_list = []
    for x in range(generations):
        children=[] 
        for i in range(0,population_size,2):
            p1,p2 = _generate_selection(pop),_generate_selection(pop)
            c1,c2 = _generate_crossover(p1,p2,.5,get_fitness)
            c1 = _mutate(c1,ids,get_fitness,.25)
            c2 = _mutate(c2,ids,get_fitness,.18)
            children.append(c1)
            children.append(c2)
        for z in children:
            if best_eval < (z.fitness):
                print (z.fitness)
                best_eval = z.fitness
                a = [Values(food_dictionary.get(x)) for x in z.genes]
                items = [{"Item":i.name,"Calories":i.calories,"Protein":i.protein,"Sodium":i.sodium,"Carbohydrate":i.carbohydrate,"Fats":i.fat,"Generation":x} for i in a]
                print(items)
                c = Chromosome(z.genes,z.fitness)
                grocery_list.append(items)
        pop = children
        max_fitness = max([x.fitness for x in children])
        metrics.append([{'generation':x,'max_fitness':max_fitness}])
    print (metrics)
    return c,children,grocery_list
