from ch05_rbf_pca import rbf_kernel
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(7,3))

X,y = make_circles(n_samples=1000,random_state=123,noise=0.1,factor=0.2)
ax[0][0].scatter(X[y==0,0],X[y==0,1],color='red',marker='^',alpha=0.5)
ax[0][0].scatter(X[y==1,0],X[y==1,1],color='blue',marker='o',alpha=0.5)

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
ax[0][1].scatter(X_spca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[0][1].scatter(X_spca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)

ax[0][0].set_xlabel('PC1')
ax[0][0].set_ylabel('PC2')
ax[0][1].set_ylim([-1,1])
ax[0][1].set_xlabel('PC1')

X_kpca,eigX = rbf_kernel(X,gamma=15,n_components=2)
ax[1][0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color='red',marker='^',alpha=0.5)
ax[1][0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],color='blue',marker='o',alpha=0.5)


ax[1][1].scatter(X_kpca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1][1].scatter(X_kpca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)

ax[1][0].set_xlabel('PC1')
ax[1][0].set_ylabel('PC2')
ax[1][1].set_ylim([-1,1])
ax[1][1].set_xlabel('PC1')

plt.show()


