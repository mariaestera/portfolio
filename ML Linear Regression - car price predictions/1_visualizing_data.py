import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

from functions import show_frame, visualize_barplot, visualize_scatter

data = pd.read_csv("C:\\Users\\piecz\\Desktop\\do usunięcia -duże pliki\\cars predictions\\train.csv",index_col=0)
print(data.head())

#from engine we can extract information about HorsePower, number of cylinders and volume of engine
#informations about fuel_type are stored in row before, but sometimes we can complete the missing data in 'fuel_type' using 'engine'

data.loc[data['engine'].str.contains('Electric', case=False, na=False) & data['fuel_type'].isna(), 'fuel_type'] = 'Electric'
filtered_data = data[data['engine'].str.contains('Electric', case=False, na=False)]

#checking
# ...  show_frame(filtered_data,n=20)
print(data.fuel_type.unique())

#extracting informations from engine
import re
engine = data['engine']
engine_capacities = []
cylinder_counts = []
horsepower = []

hp_pattern = re.compile(r'(\d+\.\d+)HP')
capacity_pattern = re.compile(r'(\d+\.\d+)L')
cylinder_pattern = re.compile(r'(\d+) Cylinder')
for entry in engine:
    hp_match = hp_pattern.search(entry)
    if hp_match:
        horsepower.append(int(float(hp_match.group(1))))
    else:
        horsepower.append(0)

    capacity_match = capacity_pattern.search(entry)
    if capacity_match:
        engine_capacities.append(float(capacity_match.group(1)))
    else:
        engine_capacities.append(0)

    cylinder_match = cylinder_pattern.search(entry)
    if cylinder_match:
        cylinder_counts.append(int(cylinder_match.group(1)))
    else:
        cylinder_counts.append(0)

data['engine_capacities'] = engine_capacities
data['cylinder_counts'] = cylinder_counts
data['horsepower'] = horsepower
data = data.drop('engine',axis = 1)

# ... show_frame(data,n=30)
# maybe it's worth to replace 0 by mean?
# ...
#visualize_scatter('engine_capacities',data)
#visualize_scatter('cylinder_counts',data)
#visualize_scatter('horsepower',data)

for label in ['engine_capacities','cylinder_counts','horsepower']:
    mean = round(data.loc[data[label] > 0, label].mean(),1)
    data[label] = data[label].replace({0.0: mean,0:mean})


#about fuel_type: whats connections are between inforamation in fuel_type and price? it's nan,'-','not supported' the same?

data['fuel_type']= data['fuel_type'].fillna('naan')
visualize_barplot('fuel_type',data)
# '_' and 'not supported' looks very messy! but 'naan' carries information
data['fuel_type'] = data['fuel_type'].replace({'–': 'not supported'})
# do OneHotEncodingFor fuel_type
dummies = pd.get_dummies(data['fuel_type'], prefix='fuel_type')
data = pd.concat([data, dummies], axis=1)
# we can get extra information about mean price for each type of fuel, but after train_test_split so we don't drop "fuel_type" column
#what about transmission, accident, clean_title?
print(data['clean_title'].unique())
data['clean_title'] = data['clean_title'].fillna(0).replace({'Yes':1})

data['accident'] = data['accident'].fillna('naan')
visualize_barplot('accident',data)
#it's difference between naan and none reported!
dummies = pd.get_dummies(data['accident'], prefix='accident')
data = pd.concat([data, dummies], axis=1)
data = data.drop('accident',axis=1)

# let's exstract some information from transmission data!
gear_count = []
transmission_type = []
gear_pattern = re.compile(r'(\d+)-?Speed')

for entry in data['transmission']:
    gear_match = gear_pattern.search(entry)
    if gear_match:
        gear_count.append(int(gear_match.group(1)))
    else:
        gear_count.append(0)



    if 'dual' in ''.join(entry).lower():
        transmission_type.append(3)
    elif 'M/T' in entry or 'Manual' in entry or 'Mt' in entry or 'M' in entry:
        transmission_type.append(1)
    elif 'A/T' in entry or 'Automatic' in entry or 'CVT' in entry or 'At' in entry:
        transmission_type.append(2)
    else:
        transmission_type.append(0)

data['gear_count'] =gear_count
data['transmission_type']=transmission_type
data = data.drop('transmission',axis=1)

#it's good to replace 0 by the mean?
visualize_scatter('gear_count',data)
visualize_scatter('transmission_type',data)
# for transmission_type we need to leave 0 in place (it can be electric vehicle) for gear_count - similar

# now takelook at outliers - it's very expensive cars that interesting us very much
outliers = data[data['price'] > data['price'].quantile(0.95)]
# ... show_frame(outliers,n=30)
data['outliers'] = (data['price'] > data['price'].quantile(0.8)).astype(int)
show_frame(data,n=30)

# with other features we will work after train-test split

data.to_csv('data_first_preprocessing.csv')