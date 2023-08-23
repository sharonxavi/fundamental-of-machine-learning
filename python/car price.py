import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Importing the dataset
data = pd.read_csv("CarPrice.csv")

# Removing non-numeric columns like "CarName"
data = data.drop("CarName", axis=1)

# Data Exploration
print(data.head())
print(data.shape)
print(data.isnull().sum())  # Checking if the dataset has NULL Values
print(data.info())
print(data.describe())

# Correlation Analysis and Heatmap
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
correlations = data[numeric_columns].corr()

plt.figure(figsize=(20, 15))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

# Training a Car Price Prediction Model
predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

mae = mean_absolute_error(ytest, predictions)
