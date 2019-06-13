import numpy as np
import pandas as pd


from nltk.tokenize import TweetTokenizer


from collections import Counter
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',300)




#EDA

df_train = pd.read_csv('dataset/train.tsv', sep="\t")
df_test = pd.read_csv('dataset/test.tsv', sep="\t")
df_sub = pd.read_csv('dataset/sampleSubmission.csv')


#Preview
df_train.head(5)


#check null
df_train= df_train[df_train['Phrase'].str.len() >0]
df_train[df_train['Phrase'].str.len() == 0].head()


#Most common trigrams for positive reviews
text = ' '.join(df_train.loc[df_train.Sentiment == 4, 'Phrase'].values)
text_trigrams = [i for i in ngrams(text.split(), 3)]

Counter(text_trigrams).most_common(20)




#Stopwords are not useful
tokenizer = TweetTokenizer()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(df_train['Phrase'].values) + list(df_test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(df_train['Phrase'])
test_vectorized = vectorizer.transform(df_test['Phrase'])

y = df_train['Sentiment']

#Logistic regression chosen
logreg = LogisticRegression()

ovr = OneVsRestClassifier(logreg)
ovr.fit(train_vectorized, y)

scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))



'''
word-embeddings vs classical methods 
word-embeddings is superior to classical methods such as one-hot encoding, since even nuances in speech may be recognised. 
An example would be "super bad movie". "Super" is positive, while "bad" is negative. 
In classical approaches this entire phrase would be neutral, although it is really negative.
'''