from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from src.classes import Values
from src.evolution import _generate_parent,_generate_population,_generate_selection,_mutate,_generate_crossover,Chromosome
from src.util import filter_dataframe
import random
import pandas as pd
import numpy as np

# Read USDA FOOD ITEM FILE
df = pd.read_csv('data/nutrients.csv')
# Filter out certain categories
food_groups = list(set(df['group'].tolist()))
options = st.multiselect(
    'What food groups do you need?',
   food_groups)

st.write('You selected:', options)
# st.dataframe(filter_dataframe(df))
df = df[df['group'].isin(options)]
st.dataframe(df)
food_dictionary = df.set_index('id').T.to_dict()
ids = list(set(df['id']))


def null(x):
    return np.where(np.isnan(x),0.0,x)


def nutrition(meals):
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
    m = nutrition(x)
    # return (np.sum(p)*4.0-np.sum(s))/np.sum(c)
    sigma = np.sum(m,axis=0)
    # sugar_constraint = np.where(sigma[2]>0,1,0)
    # print (sugar_constraint)
    return sigma[0]*4.0/sigma[1]
    


def genetic_algorithm(generations,population_size):
    #initial population
    pop = _generate_population(10,ids,get_fitness,1000)
    best_eval = pop[0].fitness
    metrics = []
    data = []
    fitness = []
    for x in range(generations):
        children=[] 
        for i in range(0,population_size,2):
            p1,p2 = _generate_selection(pop),_generate_selection(pop)
            c1,c2 = _generate_crossover(p1,p2,.5,get_fitness)
            c1 = _mutate(c1,ids,get_fitness,.25)
            c2 = _mutate(c2,ids,get_fitness,.18)
            children.append(c1)
            children.append(c2)
        for c,z in enumerate(children):
            if best_eval < (z.fitness):
                print (z.fitness)
                best_eval = z.fitness
                a = [Values(food_dictionary.get(x)) for x in z.genes]
                for ii in z.genes:
                    nutrition = Values(food_dictionary.get(ii))
                    data.append({"name":nutrition.name,"calories":nutrition.calories,"protein":nutrition.protein,"generation":"generation {0}".format(x)})
                
                # print ([i.name for i in a])
                c = Chromosome(z.genes,z.fitness)
            fitness.append({"generation":x,"children":"child {0}".format(c),"Fitness":z.fitness})
        pop = children
        max_fitness = max([x.fitness for x in children])
        metrics.append({'generation':'generation {0}'.format(x),'max_fitness':max_fitness})
    # print(df)    
    print (metrics)
    df = pd.DataFrame(data)
    return c,children,df,metrics,fitness



c,children,df,metrics,fitness = (genetic_algorithm(100,100))
#Creating the DataFrames
metrics = pd.DataFrame(metrics)
fitness = pd.DataFrame(fitness)
df2 = fitness.groupby('generation').agg({'Fitness':max}).sort_values(['Fitness','generation'],ascending=False).head(10)
df2 = df2.sort_values('generation')
st.title('Diet Optimization')
st.dataframe(df)  # Same as st.write(df)
# st.dataframe(metrics)
st.dataframe(df2) 
st.line_chart(df2)
st.text('This is a web app to allow exploration of Genetic Algorithm for a high density')

