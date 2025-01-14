#import the packages
import pandas as pd
import numpy as np

#Read dataset
dataset=pd.read_csv("kdata.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

#import KNeighborshood Classifier and create object of it
from sklearn.neighbors import KNeighborsClassifier

#Creating model
classifier=KNeighborsClassifier(n_neighbors=3)
# Training model
classifier.fit(X,y)

#predict the class for the point(6,6)
X_test=np.array([6,6])
# Predictions for test data
y_pred=classifier.predict([X_test])
print(y_pred)

# KNeighborsClassifier looks for the 5 nearest neighbors
#If set to uniform, all points in each neighbourhood have 
#equal influence in predicting class i.e. predicted class is the class with highest number of points in the neighbourhood.
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)

#predict the class for the point(6,6)
X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print(y_pred)





