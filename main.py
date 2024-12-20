#Imort Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization

# Load the data
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # Column names
df = pd.read_csv('iris.data', names = columns) 
df.head()
