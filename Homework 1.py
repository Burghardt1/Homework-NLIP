import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


dataframe=pd.read_csv("dataset_simple.csv")
dataframe.head(10)

dataframe.describe()

dataframe.shape
dataframe['label'].value_counts()

x_matrix = dataframe.copy()
x_matrix.drop(['label'], axis=1, inplace=True)
y_vector = dataframe['label']

sc = StandardScaler()
sc.fit(x_matrix)
x_matrix = sc.transform(x_matrix)

############feature selection

classif = ExtraTreesClassifier(n_estimators=100)
classif = classif.fit(x_matrix, y_vector)
classif.feature_importances_  
selected = SelectFromModel(classif, prefit=True)
x_matrix_new = selected.transform(x_matrix)
x_matrix_new.shape

X_train, X_test, y_train, y_test = train_test_split(x_matrix_new, y_vector, test_size=0.33, random_state=42,shuffle=True)

#############



classif_LR = LogisticRegression()
classif_KN = KNeighborsClassifier()
classif_RF = RandomForestClassifier()

###############Log
    
classif_LR.fit(X_train, y_train)

#############KNeighbours

classif_KN.fit(X_train, y_train)

############Random Forest

classif_RF.fit(X_train, y_train)

###########Cross validation


#Loglikelihood
scores_LR = cross_validate(estimator=classif_LR, X=X_train, y=y_train, cv=10, scoring='roc_auc')

print('AUC Loglikelihood: ', scores_LR['test_score'])
print('Mean AUC: ', np.mean(scores_LR['test_score']))
print("")

#KNeighbours
scores_KN = cross_validate(estimator=classif_KN, X=X_train, y=y_train, cv=10, scoring='roc_auc')

print('AUC KNeighbours: ', scores_KN['test_score'])
print('Mean AUC: ', np.mean(scores_KN['test_score']))
print("")

#Random Forest
scores_RF = cross_validate(estimator=classif_RF, X=X_train, y=y_train, cv=10, scoring='roc_auc')

print('AUC Random Forest: ', scores_RF['test_score'])
print('Mean AUC: ', np.mean(scores_RF['test_score']))
print("")

