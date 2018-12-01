import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


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

label = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
         0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#chi2 feature selection
number_of_features = 20
selector = SelectKBest(chi2, k=number_of_features)
X_new = selector.fit_transform(relMatrix, label)
print(X_new.shape)
print(selector.get_support(indices=True))
X = X_new
y = label

random_state = np.random.RandomState(0)

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(C=1.0, cache_size=2000, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovo', gamma='auto', kernel='linear',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

tprs = []
aucs = []
resolution = 100
mean_fpr = np.linspace(0, 1, resolution)

#fig = plt.figure(dpi=200)
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.scatter((np.where(mean_tpr==1)[0][0])/resolution,1,marker='*', color='r',s=200)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC8hr'+str(number_of_features)+'.png',dpi=200)
plt.show()
