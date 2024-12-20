# Import Packages
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

# Separate features and target because we need to feed the features to the model to make predictions
data = df.values # Convert the dataframe to a numpy array
X = data[:, 0:4] # Features are the first 4 columns (Sepal length, Sepal width, Petal length, Petal width)
y = data[:, 4] # Target is the last column (Class labels)

# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

# Model training
# Split the data to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Model evaluation
#Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# Test the model with a new data
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 6.0, 2.5], [6.4, 3.2, 4.5, 1.5]]) # Should predict Setosa, Virginica, Versicolour
# Predcit the class labels
prediction = svn.predict(new_data)
print("Predictions of Species: {}".format(prediction))

# Save the model
import pickle
with open ('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)
    
# Load the model
with open ('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
    model.predict(new_data)
