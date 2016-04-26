import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

import time
import sys

#choose which size dataset and number of classes to use
X_train = pickle.load( open("../generate_train_test/500000_100classes/X_train.p",'rb'))
X_test = pickle.load( open("../generate_train_test/500000_100classes/X_test.p",'rb'))
y_train = pickle.load( open("../generate_train_test/500000_100classes/y_train.p",'rb'))
y_test = pickle.load( open("../generate_train_test/500000_100classes/y_test.p",'rb'))

num_classes = len(set(y_test))
#print num_classes

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
print "TFIDF " + str(time_elapsed)
start_time = time.time()

#list possible baseline classifiers
dt_clf = DecisionTreeClassifier(max_depth=2)
lr_clf = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial')
lr2_clf = LogisticRegression(penalty='l2',multi_class='ovr')
nb_clf = MultinomialNB()

#choose classifier
clfs = [dt_clf]

for clf in clfs:
	#if you want to calculate accuracy
	#print clf.score(X_test_tfidf,le_y_test)

	correct = 0

	clf.fit(X_train_tfidf,le_y_train)

	time_elapsed = time.time() - start_time
	print "Fitting " + str(time_elapsed)

	pred_probs = clf.predict_proba(X_test_tfidf)

	#suggest the 5 most likely classes, calculate top-5 precision
	for i in range(len(y_test)):
		probs = pred_probs[i]
		indices = np.arange(len(probs))

		probs,indices = (list(x) for x in zip(*sorted(zip(probs, indices),reverse=True)))

		k_choices = indices[:5]

		if le_y_test[i] in k_choices:
			correct+=1

	print 100.0*correct/len(y_test)
