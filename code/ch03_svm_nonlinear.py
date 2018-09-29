import numpy as np
from sklearn.svm import SVC
from ch02 import plt_dec_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor,1,-1)

# plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
# plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
# plt.legend()

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3, 3)


svm = SVC(kernel='rbf',random_state=0,gamma=1.0,C=10.0)
svm.fit(X_xor,y_xor)

ax = plt.subplot(gs[0,0])
fig = plt_dec_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('SVM gamma=1.0')

svm = SVC(kernel='rbf',random_state=0,gamma=0.1,C=10.0)
svm.fit(X_xor,y_xor)

ax = plt.subplot(gs[0,1])
fig = plt_dec_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('SVM gamma=0.1')


svm = SVC(kernel='rbf',random_state=0,gamma=10,C=10.0)
svm.fit(X_xor,y_xor)

ax = plt.subplot(gs[1,0])
fig = plt_dec_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.title('SVM gamma=10')

plt.show()