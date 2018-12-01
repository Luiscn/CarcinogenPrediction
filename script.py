import numpy as np
import matplotlib.pyplot as plt

nFeature_s = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50,100,200,500,1000,
              2000,5000,10000,15000,20000, 25000, 30000, 35000, 40000, 45000, 50000,54675]
init = np.zeros(len(nFeature_s))
recall_s, acc_s, precision_s, specificity_s = init.copy(),init.copy(),init.copy(),init.copy()

for iter_n_features in range(len(nFeature_s)):
    nFeatures = nFeature_s[iter_n_features]
    runfile('/Users/luis/Documents/CompBio/project/py/main.py', wdir='/Users/luis/Documents/CompBio/project/py')
    recall_s[iter_n_features] = recall
    acc_s[iter_n_features] = acc
    precision_s[iter_n_features] = precision
    specificity_s[iter_n_features] = specificity
  
plt.plot(nFeature_s, recall_s)
plt.plot(nFeature_s, acc_s)
plt.plot(nFeature_s, precision_s)
plt.plot(nFeature_s, specificity_s)
plt.gca().legend(('recall','accuracy', 'precision', 'specificity'))
#plt.title('res')
plt.savefig('quantitative_results.png',dpi=200)
plt.show()

plt.plot(nFeature_s, recall_s)
plt.title('recall')
plt.savefig('recall.png',dpi=200)
plt.show()