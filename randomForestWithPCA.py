import csv
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

data = [ i for i in csv.reader(file('wine.data', 'rb')) ]
random.shuffle(data)

X = np.array([ [ float(j) for j in i[1:] ] for i in data ])
Y = np.array([ int(i[0]) for i in data ])

pca = PCA(n_components=13)
pca.fit(X)

#print pca.explained_variance_ratio_

X = pca.transform(X)

K = 5

skf = StratifiedKFold(Y, K)
for train_index_vector, test_index_vector in skf:
    X_train = X[train_index_vector]
    Y_train = Y[train_index_vector]
    X_test = X[test_index_vector]
    Y_test = Y[test_index_vector]

    classifier = RandomForestClassifier(n_estimators=10)
    classifier = classifier.fit(X_train, Y_train)

    Y_predict = classifier.predict(X_test)

    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    print 'Accuracy = %s' % (float(equal)/len(Y_predict))