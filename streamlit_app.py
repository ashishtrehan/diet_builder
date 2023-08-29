from collections import namedtuple
import altair as alt
import math
import time
import pandas as pd
import streamlit as st
import json
from src.classes import Values
from src.evolution import _generate_parent,_generate_population,_generate_selection,_mutate,_generate_crossover,Chromosome
from src.util import filter_dataframe,filter_dataset
from src.app import genetic_algorithm
import random
import pandas as pd
import numpy as np


#custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Read USDA FOOD ITEM FILE
df = pd.read_csv('data/nutrients.csv')
# Filter out certain categories
food_groups = list(set(df['group'].tolist()))
options = st.multiselect(
    'What food groups do you need?',
   food_groups,default=['Breakfast Cereals','Vegetables and Vegetable Products','Beef Products'])

file = open('data/parameters.json').read()
params_dict = json.loads(file)
diets = params_dict.get('Diets')
names = [x.get('Name') for x in diets]




st.write('You selected:', options)
# st.dataframe(filter_dataframe(df))
df = df[df['group'].isin(options)]
st.dataframe(df)
food_dictionary = df.set_index('id').T.to_dict()
ids = list(set(df['id']))


with st.form('Diet Optimization'):
    with st.sidebar: 
        st.write("Choose Parameters")
        generations_val = st.slider("Generations",min_value=1, max_value=100, value=100)
        population_size = st.slider("Population Size",min_value=1, max_value=100, value=100)
        n_items = st.slider("Number of Food Items",min_value=1, max_value=10, value=3)
        st.selectbox('What kind of diet?',names)
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Generations", generations_val, "Population Size", population_size,"Number of Items", n_items)
        with st.spinner("Loading...Genetic Algorithm"):
            c,children,grocery_list = genetic_algorithm(generations_val,ids,food_dictionary,population_size,n_items)
            time.sleep(5)
            generation_0 = pd.DataFrame(grocery_list[0])
            generation_0.loc['total']= generation_0.sum()
            generation_0.loc[generation_0.index[-1], 'Item'] = ''
            total_gen_0 = generation_0[generation_0['Item'] == '']
            calories_gen_0 = int(total_gen_0['Calories'].values[0])
            protein_gen_0 = round(total_gen_0['Protein'].values[0],2)
            sodium_gen_0 = round(total_gen_0['Sodium'].values[0],2)
            carbohydrates_gen_0 = round(total_gen_0['Carbohydrate'].values[0],2)
            fats_gen_0 = round(total_gen_0['Fats'].values[0],2)
            print('Gen 0 DF {0}'.format(calories_gen_0))
            for x in grocery_list:
        # st.write("Items {0}".format(':tomato: :shrimp: :chicken: :apple: :fruit:'), x)
                df = pd.DataFrame(x).sort_values(by='Generation')
                df.loc['total']= df.sum()
                df.loc[df.index[-1], 'Item'] = ''
                
                # st.write(int(calories_gen_0.values[0]))
                col1, col2, col3, col4, col5 = st.columns(5)
                calories = df.loc[df.index[-1], 'Calories'] 
                protein = df.loc[df.index[-1], 'Protein'] 
                sodium = df.loc[df.index[-1], 'Sodium'] 
                carbohydrates = df.loc[df.index[-1], 'Carbohydrate']
                fats = df.loc[df.index[-1], 'Fats']
                col1.metric("Calories", "{0}".format(int(calories)), "{0}".format(calories - calories_gen_0))
                col2.metric("Protein", "{0}".format(protein), "{0} g".format(round(protein - protein_gen_0,2)))
                col3.metric("Sodium", "{0}".format(sodium), "{0} g".format(round(sodium - sodium_gen_0,2)))
                col4.metric("Carbohydrates", "{0}".format(carbohydrates), "{0} g".format(round(carbohydrates - carbohydrates_gen_0,2)))
                col5.metric("Fats", "{0}".format(fats), "{0} g".format(round(fats - fats_gen_0,2)))
                with st.expander("Generation {0} Food Items :apple: :chicken:".format(x[0].get('Generation'))):
                    chart_data = pd.DataFrame(np.random.randn(20, 3),columns=["a", "b", "c"])
                    st.bar_chart(chart_data)
                    st.dataframe(df)
                # with st.container():
                #     col1, col2 = st.columns(2)
                #     with col1:
                #         items = list(df['Item'])
                #     with col2:
                #         col1.metric("Temperature", "70 °F", "1.2 °F")
                #         col2.metric("Wind", "9 mph", "-8%")
                #         col3.metric("Humidity", "86%", "4%")
                time.sleep(0.5)

# c,children,df,metrics,fitness = (genetic_algorithm(100,ids,food_dictionary,100))
# #Creating the DataFrames
# metrics = pd.DataFrame(metrics)
# fitness = pd.DataFrame(fitness)
# df2 = fitness.groupby('generation').agg({'Fitness':max}).sort_values(['Fitness','generation'],ascending=False).head(10)
# df2 = df2.sort_values('generation')
# # st.dataframe(metrics)
# st.dataframe(df2) 
# st.line_chart(df2)


