import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

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

#split into train and validation
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

start_time = time.time()

#numerically encode the classes
le = LabelEncoder()
le.fit(y_train)
le_y_train = le.transform(y_train)
le_y_validate = le.transform(y_validate)
le_y_test = le.transform(y_test)

#create TFIDF features from text
tf_transformer = TfidfVectorizer(stop_words='english').fit(X_train)
X_train_tfidf = tf_transformer.transform(X_train)
X_validate_tfidf = tf_transformer.transform(X_validate)
#X_test_tfidf = tf_transformer.transform(X_test)

time_elapsed = time.time() - start_time
print "TFIDF: " + str(time_elapsed/60) + " minutes"
start_time = time.time()

train_precisions = []
validate_precisions = []

ada_clf = AdaBoostClassifier(base_estimator=LogisticRegression(penalty='l2',multi_class='ovr'),n_estimators=30)

ada_clf.fit(X_train_tfidf,le_y_train)

time_elapsed = time.time() - start_time
print "Fitting: " + str(time_elapsed/60) + " minutes"


train_pred_probs = ada_clf.predict_proba(X_train_tfidf)
val_pred_probs = ada_clf.predict_proba(X_validate_tfidf)


#suggest the k most likely classes, then calculate top-k precision for training data
correct = 0
for i in range(len(y_train)):
	k_choices = top_k_predictions(5,train_pred_probs[i])

	if le_y_train[i] in k_choices:
		correct+=1

train_precisions.append(100.0*correct/len(y_train))


#suggest the k most likely classes, then calculate top-k precision for validation data
correct = 0
for i in range(len(y_validate)):
	k_choices = top_k_predictions(5,val_pred_probs[i])

	if le_y_validate[i] in k_choices:
		correct+=1

validate_precisions.append(100.0*correct/len(y_validate))

print c_values
print train_precisions
print validate_precisions

"""
#Now plot            
fig = plt.figure()
ax=fig.add_subplot(111)

plt.plot(c_values,train_precisions,'b-',label='Training precisions')
plt.plot(c_values,validate_precisions,'r-',label = 'Validation precisions')

#set scale of x-axis to log-scale
plt.xscale('log')

#set axis labels and title
ax.set_xlabel('C (regularization parameter)')
ax.set_ylabel('Precision')
plt.legend(loc=1)
plt.title('Train/Validation Precision for Various C Values')
plt.savefig('log_reg.png')"""