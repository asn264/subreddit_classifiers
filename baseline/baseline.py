import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = pickle.load( open("X_train.p",'rb'))
X_test = pickle.load( open("X_test.p",'rb'))
y_train = pickle.load( open("y_train.p",'rb'))
y_test = pickle.load( open("y_test.p",'rb'))

tf_transformer = TfidfVectorizer(stop_words='english').fit(X_train)
X_train_tfidf = tf_transformer.transform(X_train)
X_test_tfidf = tf_transformer.transform(X_test)
