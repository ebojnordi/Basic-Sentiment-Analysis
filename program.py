# A Basic Semantic Analysis 
# Published by: Ehsan Bojnordi -- 11/11/2021
import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

############## Importing Dataset ###################
df = pd.read_excel('train.xlsx')
df.head()

############## Removing Punctuation ###################
df['new_reviews'] = df['Reviews'].str.replace('[^\w\s]','', regex=True)
df.head()

############## Removing Stop Words ###################
stop = stopwords.words('english')
df['new_reviews'].apply(lambda x: [item for item in str(x).split() if item not in stop])
df.head(20)

############## Transforming words into lower case ###################
df['new_reviews'] = df['new_reviews'].str.lower().replace('\\', '').replace('_', ' ')
df.head()

############## Lemmatizing Words ###################
lmtzr = WordNetLemmatizer()
df['new_reviews'].apply(lambda lst:[lmtzr.lemmatize(word) for word in str(lst).split()])
df.head()

############## Vectorizing Step ###################
tfidf=TfidfVectorizer(max_features=5000)
X=df['new_reviews']
y=df['Sentiment']
X=tfidf.fit_transform(X.values.astype('U'))

############## Setting Classifying Parameters ###################
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

############## Classification Report ###################
print(classification_report(y_test, y_pred))

S= 'What an awesome movie!!! I really recommend it.'

vec=tfidf.transform([S])
clf.predict(vec)
