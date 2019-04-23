import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import scipy as sp

colnames = ['label', 'sms_message']
df = pd.read_csv('SMSSpamCollection', sep='\t', names=colnames)
df.head(n=5)
df.groupby('label').describe()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})



print(df["label"].value_counts())
print(df.shape)
print(df.groupby('label').describe())


X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=11)


#no stop words because very short 
count_vector = CountVectorizer() #stop words not used because SMS messages are very short
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

#######

multi_NB = MultinomialNB()
multi_NB.fit(training_data, y_train)

print(classification_report(y_test, multi_NB.predict(testing_data)))




prob_spam = sum(y_train) / len(y_train)
prob_ham = 1 - prob_spam


####prob ham
ham_loc = np.where(y_train == 0)
ham = training_data.tocsr()[ham_loc]

ham_freq = ham.toarray().sum(axis=0)+1
prob_ham2 = ham_freq / (sum(ham_freq))

#####prob spam
spam_loc = np.where(y_train == 1)
spam = training_data.tocsr()[spam_loc]

spam_freq = spam.toarray().sum(axis=0)+1
prob_spam2 = spam_freq / (sum(spam_freq))



report = []


def decider(key):
    val = sp.sparse.find(key)
    prob_ham3 = np.log(prob_ham)
    prob_spam3 = np.log(prob_spam)
    
    
    for a in range(len(val[1])):
        prob_ham3 =+ np.log(prob_ham2[val[1][a]])*val[2][a]
        prob_spam3 =+ np.log(prob_spam2[val[1][a]])*val[2][a]

    if prob_spam3 >= prob_ham3:
        return 1
    else:
        return 0


for a in testing_data:
    report.append(decider(a))
    

print(classification_report(y_test, report))
