import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

os.chdir('/Users/luis/Dropbox/CompBio')
data = sio.loadmat('TGGATES')

os.chdir('/Users/luis/Documents/CompBio/project/py')
ctrlMatrix = data['ctrlMatrix']
highMatrix = data['highMatrix']

relMatrix = highMatrix / ctrlMatrix
relMatrix_store = relMatrix
relMatrix = relMatrix_store

relMatrix = sklearn.preprocessing.normalize(relMatrix, norm='l2', axis=0, copy=True, return_norm=False)

print(relMatrix.shape)

true = np.array(['N', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'N', 'Y', 'Y',
        'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'N', 'Y', 'N', 'N',
        'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N',
        'N', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'Y', 'N',
        'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 
        'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'Y',
        'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'N',
        'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N',
        'Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y',
        'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])

label = np.zeros(150)
for i in range(150):
    if true[i] == 'N':
        label[i] = 0
    elif true[i] == 'Y':
        label[i] = 1

#chi2 feature selection
selector = SelectKBest(chi2, k=5000)
X_new = selector.fit_transform(relMatrix, label)
print(X_new.shape)
#print(selector.get_support(indices=True))
relMatrix = X_new

# Binarize the output
y = label
n_classes = 2

X = relMatrix
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = svm.SVC(C=1.0, cache_size=2000, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovo', gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])