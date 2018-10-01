import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from ch02 import plot_decision_regions

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wine/wine.data', header=None)

X,y = df.iloc[:,1:].values, df.iloc[:,0].values 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# cov_mat = np.cov(X_train_std.T)
# eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
# print("Eigen values\n %s" % eigen_vals)

# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
# cum_var_exp = np.cumsum(var_exp)


# plt.bar(range(1,14), var_exp,alpha=0.5,align='center',label='individual explained variance')
# plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.show()

# eig_pairs = [(np.abs(eigen_vals[i]),np.abs(eigen_vecs[:,i])) for i in range(len(eigen_vals))]
# print(eig_pairs[0])
# eig_pairs.sort(reverse=True)
# print('\n', eig_pairs[0])

# w = np.hstack((eig_pairs[0][1][:,np.newaxis],eig_pairs[1][1][:,np.newaxis]))
# np.set_printoptions(suppress=True)
# print('Matrix W: \n',w)

# X_train_pca = X_train_std.dot(w)
# X_test_pca = X_test_std.dot(w)

# colors = ['r','b','g']
# markers = ['s','x','o']

# for l,c,m in zip(np.unique(y_train),colors,markers):
#     plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],c=c,label=l,marker=m)

# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = -1*pca.fit_transform(X_train_std)
X_test_pca = -1*pca.transform(X_test_std)

lr.fit(X_train_pca,y_train)
p=plot_decision_regions(X_train_pca,y_train,classifier=lr)
p.xlabel('PC1')
p.ylabel('PC2')
p.legend(loc='lower right')
p.show()

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print( pca.explained_variance_ratio_)