import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize


os.chdir('/Users/luis/Dropbox/CompBio')
data = sio.loadmat('TGGATES8hr')

os.chdir('/Users/luis/Documents/CompBio/project/py')
ctrlMatrix = data['ctrlMatrix']
highMatrix = data['highMatrix']

relMatrix = highMatrix / ctrlMatrix
relMatrix_store = relMatrix
relMatrix = relMatrix_store

relMatrix = sklearn.preprocessing.normalize(relMatrix, norm='l2', axis=0, copy=True, return_norm=False)

print(relMatrix.shape)

label = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                  0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0])

#chi2 feature selection
selector = SelectKBest(chi2, k=nFeatures)
#selector = SelectKBest(chi2, k=5000)
X_new = selector.fit_transform(relMatrix, label)
print(X_new.shape)
#print(selector.get_support(indices=True))
relMatrix = X_new

loo = LeaveOneOut()
loo.get_n_splits(relMatrix)

TP, TN, FP, FN = 0,0,0,0
for train_index, test_index in loo.split(relMatrix):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    print("TEST:", test_index)
    X_train, X_test = relMatrix[train_index], relMatrix[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    clf = SVC(C=1.0, cache_size=2000, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovo', gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X_train, y_train) 
    pred = clf.predict(relMatrix[test_index,:])
    GT = label[test_index]
    
    if pred == 1 and GT == 1:
        TP += 1
    elif pred == 1 and GT == 0:
        FP += 1
    elif pred == 0 and GT == 1:
        FN += 1
    elif pred == 0 and GT == 0:
        TN += 1
    
#    print(str(pred) + str(GT))
    if pred != GT:
        print(test_index)
print('pred, true')
acc = (TP + TN) / len(label)
precision = TP / np.max(((TP + FP), 1e-3))
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
print('acc = ' + str(acc))
print('pred''\t'+ 'actual')
print('\t'+'pos'+'\t'+'neg')
print('pos'+'\t'+str(TP)+'\t'+str(FP))
print('neg'+'\t'+str(FN)+'\t'+str(TN))
print('recall = ' + str(recall))


