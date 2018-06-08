import csv
import random

import datetime

startTime = datetime.datetime.now()
data = [i for i in csv.reader(file('wine.data', 'rb'))]
random.shuffle(data)
X = [i[1:] for i in data]
Y = [i[0] for i in data]

train_cutoff = len(data) * 3/4

X_train = X[:train_cutoff]
Y_train = Y[:train_cutoff]
X_test = X[train_cutoff:]
Y_test = Y[train_cutoff:]

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10)
classifier = classifier.fit(X_train, Y_train)
Y_predict = classifier.predict(X_test)


def accuracy(predict, test):
    equal = 0
    for i in xrange(len(predict)):
        if predict[i] == test[i]:
            equal += 1
    return float(equal) / len(predict)


for i in xrange(len(Y_predict)):
    print ' %s : %s ' % (Y_predict[i], Y_test[i])
    pass

endTime = datetime.datetime.now()
print 'total test = %d' % len(Y_test)
print 'Accuracy = %s' % (accuracy(Y_predict, Y_test))
print 'Runtime = %d' % (endTime - startTime).microseconds
