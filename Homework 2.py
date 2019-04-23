import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import scipy


colnames = ['label', 'sms_message']
df = pd.read_csv("SMSSpamCollection", sep='\t', names=colnames)
df['label'] = df.label.map({'ham':0, 'spam':1})

print(df.head(5))
print(df.groupby('label').describe())
#imbalanced
print(df.shape)
#no missing values
print(df.isnull().values.any())


X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=11)

#stop words ignored since very small messages 
count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# Number of unique word entries
print(len(count_vector.get_feature_names()))




####Naive bayes (independence assumed)

prob_spam = sum(y_train) / len(y_train)
prob_ham = 1 - prob_spam


### Prob spam


inst_spam = np.where(y_train==1)[0]
spam = training_data.tocsr()[inst_spam,:]

frequency_spam = spam.toarray().sum(axis=0) + 1
probability_spam = frequency_spam / (sum(frequency_spam))

'''

### Prob ham 

indices = np.where(y_train == 0)[0]
ham = training_data.tocsr()[indices,:]

frequency_ham = ham.toarray().sum(axis=0) + 1
probability_ham = frequency_ham / (sum(frequency_ham))

#### P(ham/text) and p(spam/text)

def spam_or_ham(arr):
    prob_ham2 = np.log(prob_ham)
    prob_spam2 = np.log(prob_ham)
    arr = scipy.sparse.find(arr)
    for i in range(len(arr[1])):
        prob_ham2 = prob_ham2 + np.log(probability_ham[arr[1][i]]) * arr[2][i]
        prob_spam2 = prob_spam2 + np.log(probability_spam[arr[1][i]]) * arr[2][i]

    if prob_ham >= prob_spam:
        return 0
    else:
        return 1

ans = []
for i in training_data:
    ans.append(spam_or_ham(i))

    

#### Accuracy
    
from sklearn.metrics import classification_report

#print(classification_report(y_test, ans))



#### built in solution
# training model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)


print(classification_report(y_test, naive_bayes.predict(training_data)))

'''