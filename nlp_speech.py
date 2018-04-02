#!usr/bin/python3

import nltk
from nltk.corpus import stopwords
import speech as sp
from nltk.tokenize import sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm, metrics
from sklearn.svm import SVC
from textblob import TextBlob


#nltk.download() #Run this step if you do not have the NLTK packages. 

#Removing STOP Words
stop_words = stopwords.words('english')
tokens = [ j for i in sp.transcript_strings for j in i.split()]

clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

word_matcher = {}
def transcript_word_analysis2():
	for j in clean_tokens:
			j = j.lower()
			if j in word_matcher:
				word_matcher[j]+=1
			else:
				word_matcher[j]=1

	word_series = pd.Series(word_matcher,name="Word Counter")
	word_series.plot.bar()
	plt.title("Word Distribution After Removing STOPWORDS")
	plt.show()
transcript_word_analysis2()

def get_phrase_sentiment(phrase):
    analysis = TextBlob(phrase)
    if analysis.sentiment.polarity > 0:
        return 'positive', float(format(analysis.sentiment.polarity, '.3f'))
    elif analysis.sentiment.polarity == 0:
        return 'neutral', float(format(analysis.sentiment.polarity, '.3f'))
    else:
        return 'negative', float(format(analysis.sentiment.polarity, '.3f'))

transcript_str_string = " ".join(list(word_matcher))
string_list = transcript_str_string.split()
st_dict = {}
for i in string_list:
    if not i[-1].isalpha():
        i = i[:-1]
    st_dict[i] = get_phrase_sentiment(i)


df = pd.DataFrame.from_dict(st_dict,orient="index")
df.reset_index()
df.columns = ["polarity_nature","polarity_value"]
print(df.head())

def new_column(j):
    if j in ["neutral","positive"]:
        return ({"fair":j},1)
    else:
        return ({"unfair":j},0)
df["status"] = df.polarity_nature.apply(new_column)
my_list = df["status"].values

neg_features = []
pos_features = []
for k in my_list:
    if list(k[0])=="fair":
        pos_features.append(k)
    else:
        neg_features.append(k)

negcutoff = len(neg_features)*3//4
poscutoff = len(pos_features)*3//4
trainfeats = neg_features[:negcutoff] + pos_features[:poscutoff]
testfeats = neg_features[negcutoff:] + pos_features[poscutoff:]
print ('\n')
print('Total Training Instances - '+ str(len(trainfeats)))
print( 'Total Testing Instances - ' + str(len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print ('\n')
print('NaiveBayesClassifier accuracy:', nltk.classify.util.accuracy(classifier, testfeats))


classifier1 = DecisionTreeClassifier.train(trainfeats,entropy_cutoff=0)
print ('\n')
print('DecisionTreeClassifier accuracy:', nltk.classify.util.accuracy(classifier1, testfeats))

feature_names = ["polarity_nature","polarity_value"]
X = df[feature_names]
X.polarity_nature = X.polarity_nature.apply(lambda i: 0.0 if i=="neutral" else ( 1.0 if i=="postive" else -1.0))
df["status1"] = df.status.apply(lambda i: 0.0 if i==({u'fair': u'neutral'}, 1) else ( 1.0 if i==({u'fair': u'positive'}, 1) else -1.0))
y = df.status1
print (y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.2)
print (len(X_train), len(X_test))


linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print("\n")
print (y_pred)
print (y_pred.round())
print (y_test)
print ("Linear Regression Accuracy -", metrics.accuracy_score(y_test, y_pred.round()))

clf = svm.SVC()
clf.fit(X_train, y_train)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
y_pred1 = clf.predict(X_test)
print("\n")
print (y_pred1)
print ("SVC Accuracy -", metrics.accuracy_score(y_test, y_pred1.round()))

if __name__ == "__main__":
	pass
