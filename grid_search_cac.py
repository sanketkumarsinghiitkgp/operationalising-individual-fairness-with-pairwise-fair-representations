"""
Created on Sat Feb 29 17:29:18 2020

@author: Harshvardhan
"""
import numpy as np
import pandas as pd

path = './Desktop/termproject/'

df_crime_dataset = pd.read_csv(path + 'communities.data', na_values='?', header=None)

headers_df = pd.read_csv('communities.names', sep=' ', header=None)
df_crime_dataset.columns = headers_df[1].tolist()

df_crime_dataset.describe()
# set nan to zero
df_crime_dataset.fillna(value=0, inplace=True)
# shuffling
df_crime_dataset = df_crime_dataset.sample(frac=1).reset_index(drop=True)

#Code for dimensionality reduction and training and test split
'''

'''

#Grid_Search
from sklearn.model_selection import GridSearchCV
clf = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics 
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

#Logistic Regression (Grid Search) Confusion matrix
confusion_matrix(y_test,y_pred_acc)
