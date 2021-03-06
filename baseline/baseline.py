import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

import time
import sys

def top_k_predictions(k,probs):
	indices = []
	for i in range(k):
		index = np.argmax(probs)
		indices.append(index)
		probs[index] = 0

	return indices

#choose which size dataset and number of classes to use
X_train = pickle.load( open("../generate_train_test/final_dataset/X_train.p",'rb'))
X_test = pickle.load( open("../generate_train_test/final_dataset/X_test.p",'rb'))
y_train = pickle.load( open("../generate_train_test/final_dataset/y_train.p",'rb'))
y_test = pickle.load( open("../generate_train_test/final_dataset/y_test.p",'rb'))

num_classes = len(set(y_test))

start_time = time.time()

#numerically encode the classes
le = LabelEncoder()
le.fit(y_train)
le_y_train = le.transform(y_train)
le_y_test = le.transform(y_test)

#create TFIDF features from text
tf_transformer = TfidfVectorizer(stop_words='english').fit(X_train)
X_train_tfidf = tf_transformer.transform(X_train)
X_test_tfidf = tf_transformer.transform(X_test)

time_elapsed = time.time() - start_time
print "TFIDF: " + str(time_elapsed/60) + " minutes"
start_time = time.time()

#list possible baseline classifiers
dt_clf = DecisionTreeClassifier(max_depth=2)
dt_ovr_clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=2))
lr_clf = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial')
lr2_clf = LogisticRegression(penalty='l2',multi_class='ovr')
nb_clf = MultinomialNB()
#choose classifier
clfs = [lr2_clf,nb_clf,dt_clf]

train_precisions = []
test_precisions = []

for clf in clfs:

	clf.fit(X_train_tfidf,le_y_train)

	time_elapsed = time.time() - start_time
	print "Fitting: " + str(time_elapsed/60) + " minutes"

	train_pred_probs = clf.predict_proba(X_train_tfidf)
	test_pred_probs = clf.predict_proba(X_test_tfidf)

	#suggest the k most likely classes, then calculate top-k precision for training data
	correct = 0
	for i in range(len(y_train)):
		k_choices = top_k_predictions(5,train_pred_probs[i])

		if le_y_train[i] in k_choices:
			correct+=1

	train_precisions.append(100.0*correct/len(y_train))

	#suggest the 5 most likely classes, calculate top-5 precision for test data
	correct = 0
	for i in range(len(y_test)):
		k_choices = top_k_predictions(5,test_pred_probs[i])

		if le_y_test[i] in k_choices:
			correct+=1

	test_precisions.append(100.0*correct/len(y_test))

print train_precisions
print test_precisions
