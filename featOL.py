import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

os.chdir('/Users/luis/Dropbox/CompBio')
data24 = sio.loadmat('TGGATES')

os.chdir('/Users/luis/Documents/CompBio/project/py')
ctrlMatrix24 = data24['ctrlMatrix']
highMatrix24 = data24['highMatrix']

relMatrix24 = highMatrix24 / ctrlMatrix24

relMatrix24 = sklearn.preprocessing.normalize(relMatrix24, norm='l2', axis=0, copy=True, return_norm=False)

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

label24 = np.zeros(150)
for i in range(150):
    if true[i] == 'N':
        label24[i] = 0
    elif true[i] == 'Y':
        label24[i] = 1

os.chdir('/Users/luis/Dropbox/CompBio')
data8 = sio.loadmat('TGGATES8hr')

os.chdir('/Users/luis/Documents/CompBio/project/py')
ctrlMatrix8 = data8['ctrlMatrix']
highMatrix8 = data8['highMatrix']

relMatrix8 = highMatrix8 / ctrlMatrix8

relMatrix8 = sklearn.preprocessing.normalize(relMatrix8, norm='l2', axis=0, copy=True, return_norm=False)

label8 = np.array([0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                  0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0])
    
k_max = 10000
listOL = np.zeros(k_max)
k_list = range(1, k_max, round(k_max/500))
for nFea in k_list: 
    selector = SelectKBest(chi2, k=nFea)
    X24 = selector.fit_transform(relMatrix24, label24)
    fea24 = list(selector.get_support(indices=True))
    
    selector = SelectKBest(chi2, k=nFea)
    X8 = selector.fit_transform(relMatrix8, label8)
    fea8 = list(selector.get_support(indices=True))

    listOL[nFea] = len(intersection(fea24, fea8))

base = np.array([t ** 2 for t in k_list])
plt.plot(k_list,base /54675, linestyle = '--', color = 'r')
plt.plot(k_list,listOL[k_list], color = None)
plt.gca().legend(('random vs. random', '8hr vs. 24hr'))
plt.title('# gene intersection')
plt.xlabel('# gene selected in the two controls')
plt.ylabel('# gene intersection')
plt.savefig('intersection'+str(k_max)+'.png',dpi=200)
plt.show()


