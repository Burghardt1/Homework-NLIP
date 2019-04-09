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

clf = ExtraTreesClassifier(n_estimators=100)
clf = clf.fit(x_matrix, y_vector)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
x_matrix_new = model.transform(x_matrix)
x_matrix_new.shape

#############

X_train, X_test, y_train, y_test = train_test_split(x_matrix_new, y_vector, test_size=0.33, random_state=42,shuffle=True)

classif_LR = LogisticRegression()
classif_KN = KNeighborsClassifier()
classif_RF = RandomForestClassifier()

###############Log
    
classif_LR.fit(X_train, y_train)
y_pred_LR= classif_LR.predict(X_test)

#############KNeighbours

classif_KN.fit(X_train, y_train)
y_pred_DT= classif_KN.predict(X_test)


############Random Forest

classif_RF.fit(X_train, y_train)
y_pred_RF= classif_RF.predict(X_test)


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

