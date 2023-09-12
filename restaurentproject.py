import numpy as np
import pandas as pd

data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(data.head())

print(data.shape)

print(data.columns)

print(data.info)

#Data Preprocessinng

import nltk
import re
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub(pattern='[^a-zA-Z]',repl=" ",string=data["Review"][i])

#print(review)
    review=review.lower()
    review_words=review.split()
    review_words=[word for word in review_words if not word in set(stopwords.words("english"))]

    ps=PorterStemmer()
    review=[ps.stem(word) for word in review_words]

    review=' '.join(review)
    #print(review)
    corpus.append(review)
    #print(corpus[:1500])
    

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values
#print(x)

#split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Model Training
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier

classifier=MultinomialNB()
#classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1=accuracy_score(y_test,y_pred)
score2=precision_score(y_test,y_pred)
score3=recall_score(y_test,y_pred)

print("---------------Score------------")
print("Accuracy Score : {}%".format(round(score1*100,2)))
print("Precision Score : {}%".format(round(score2*100,2)))
print("Recall Score : {}%".format(round(score3*100,2)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True,cmap="YlGnBu",xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

#hyperparameter tuning the naive bayes classfier
best_accuracy=0.0
alpha_val=0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier=MultinomialNB(alpha=i)
    #temp_classifier=RandomForestClassifier()
    temp_classifier.fit(X_train,y_train)
    temp_y_pred=temp_classifier.predict(X_test)
    score=accuracy_score(y_test,temp_y_pred)
    print("Accuracy score for alpha={} is : {}%".format(round(i,1),round(score*100,2)))
    if score>best_accuracy:
        best_accuracy=score
        alpha_val=i
print("--------------------------------------------------")
print("the accuract is {}% with alpha val as {}".format(round(best_accuracy*100,2),round(alpha_val,1)))


def predict_sentiment(sample_review):
    sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
    sample_review=sample_review.lower()
    sample_review_words=[word for word in sample_review if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in sample_review_words]
    final_review=' '.join(final_review)

    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)

#1
sample_review='the food is really Good'
if predict_sentiment(sample_review):
    print("Positive review")
else:
    print("Negative review")
    
#2
sample_review='the food was absolutely wonderful, from preparation to presentation ,very pleasing'
if predict_sentiment(sample_review):
    print("Positive review")
else:
    print("Negative review")


