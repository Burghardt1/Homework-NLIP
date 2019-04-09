import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


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
x_new = model.transform(x_matrix)
x_new.shape

#############

X_train, X_test, y_train, y_test = train_test_split(x_new, y_vector, test_size=0.3, random_state=42,shuffle=True)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics


clfr_LR = LogisticRegression()
clfr_DT = DecisionTreeClassifier()
clfr_RF = RandomForestClassifier()


def model_performance(model_name, X_tr, y_tr, y_te, y_pred):
    print('Model: ' + model_name)
    print('Test accuracy (Accuracy Score): %f'%metrics.accuracy_score(y_te, y_pred))
    print('Test accuracy (ROC AUC Score): %f'%metrics.roc_auc_score(y_te, y_pred))
    print('Test precision 0: %f'%metrics.precision_score(y_te, y_pred, pos_label=0))   
    print('Test precision 1: %f'%metrics.precision_score(y_te, y_pred, pos_label=1)) 
    print('Test recall 0: %f'%metrics.recall_score(y_te, y_pred, pos_label=0))   
    print('Test recall 1: %f'%metrics.recall_score(y_te, y_pred, pos_label=1))

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_te, y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    # making the graph
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

###############Log
    
clfr_LR.fit(X_train, y_train)

y_pred_LR= clfr_LR.predict(X_test)
model_performance('Logistic Regression', X_tr=X_train, y_tr=y_train, y_te=y_test, y_pred=y_pred_LR)

#############Decision Tree

clfr_DT.fit(X_train, y_train)
y_pred_DT= clfr_DT.predict(X_test)
model_performance('Decision tree classifier', X_tr=X_train, y_tr=y_train, y_te=y_test, y_pred=y_pred_DT)

############Random Forest

clfr_RF.fit(X_train, y_train)

y_pred_RF= clfr_RF.predict(X_test)
model_performance('Random forest classifier', X_tr=X_train, y_tr=y_train, y_te=y_test, y_pred=y_pred_RF)

###########Cross validation

from sklearn.model_selection import cross_validate
#Log
scores_LR = cross_validate(estimator=clfr_LR, X=X_train, y=y_train, cv=10, scoring='roc_auc')

print('AUC-ROS scores: ', scores_LR['test_score'])
print('Average AUC-ROC score: ', np.mean(scores_LR['test_score']))

#Random Forest
scores_RF = cross_validate(estimator=clfr_RF, X=X_train, y=y_train, cv=10, scoring='roc_auc')

print('AUC-ROS scores: ', scores_RF['test_score'])
print('Average AUC-ROC score: ', np.mean(scores_RF['test_score']))

