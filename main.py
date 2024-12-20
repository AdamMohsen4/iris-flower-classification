#Imort Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization

# Load the data
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # Column names
df = pd.read_csv('iris.data', names = columns) 
df.head()

df.describe() # Summary statistics

# Visualize the whole dataset
sns.pairplot(df, hue = 'Class_labels')

# Separate features and target
data = df.values # Convert the dataframe to a numpy array
X = data[:, 0:4] # Features are the first 4 columns (Sepal length, Sepal width, Petal length, Petal width)
y = data[:, 4] # Target is the last column (Class labels)