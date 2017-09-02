import numpy as np
import string
import csv
from sklearn import svm

#reads csv file and outputs it as a list of dicts
def readCsv(f):
    l = []
    with open(f) as csvfile:
        reader = csv.DictReader(csvfile)
        fn = reader.fieldnames
        for row in reader:
            l.append({fn[i]: row[fn[i]] for i in range(len(fn))})
        return l

#generates a subset from a set based on a fieldname and value
def genSet(s, field, val):
    l = [x for x in s if x[field] == val]
    return l

#generates an X set for the SVM
def genX(s):
    X = []
    for i in s:
        l = []
        if i:
            for k in i.keys():
                if k != "label":
                    if i[k] == '0':
                        l.append(0)
                    else:
                        l.append(1)
        X.append(l)
    return X

trainset = readCsv("train.csv")
testset = readCsv("test.csv")

X_train = genX(trainset)
y_train = [x['label'] for x in trainset]

X_test = genX(testset)


clf = svm.SVC(C = 1)
clf.fit(X_train, y_train)
train_predictions = clf.predict(X_train)

test_predictions = clf.predict(X_test)

#write predictions to file
with open("predictions_Digits_SVM.csv", 'w') as pred_file:
    pred_file.write("ImageId,Label\n")
    for i in range(1, len(testset)+1):
        ImageId = str(i)
        prediction = test_predictions[i-1]
        pred_file.write("{0},{1}\n".format(ImageId, prediction))
