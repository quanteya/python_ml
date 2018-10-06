import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# pca = PCA(n_components=2)
# lr = LogisticRegression()
# X_train_pca = -1*pca.fit_transform(X_train_std)
# X_test_pca = -1*pca.transform(X_test_std)

# lr.fit(X_train_pca,y_train)
# p=plot_decision_regions(X_train_pca,y_train,classifier=lr)
# p.xlabel('PC1')
# p.ylabel('PC2')
# p.legend(loc='lower right')
# p.show()

# pca = PCA(n_components=None)
# X_train_pca = pca.fit_transform(X_train_std)
# print( pca.explained_variance_ratio_)

np.set_printoptions(precision=4)
mean_vecs = []

for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label],axis=0))
    #print('MV %s: %s\n' %(label,mean_vecs[label-1]))

d = 13 
S_W = np.zeros((d,d))

# for label,mv in zip(range(1,4),mean_vecs):
#     class_scatter = np.zeros((d,d))
#     for row in X_train[y_train==label]:
#         row,mv = row.reshape(d,1), mv.reshape(d,1)
#         class_scatter+=(row-mv).dot((row-mv).T)
#     S_W+=class_scatter
# print('\n Within class scatter matrix: %sx%s' %(S_W.shape[0],S_W.shape[1]))

for label,mv in zip(range(1,4),mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W+=class_scatter
print('\n Scaled Within class scatter matrix: %sx%s\n' %(S_W.shape[0],S_W.shape[1]))


S_B = np.zeros((d,d))
mean_overall = np.mean(X_train_std,axis=0)
for i,mean_vec in enumerate(mean_vecs):
    n = X_train[y_train==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B += n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
print('\n Betwween-class scatter matrix: %sx%s\n' %(S_B.shape[0],S_B.shape[1]))


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs,key=lambda k: k[0],reverse=True)
print('\n Eigen values in decreasing order: \n')
for eig_val in eig_pairs:
    print('\n', eig_val[0])

tot = sum(eig_vals.real)
discr = [(i/tot) for i in sorted(eig_vals.real,reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14),discr,alpha=0.5,align='center',label='individual discriminablity')
plt.step(range(1,14),cum_discr,where='mid',label='cumulative discriminability')
plt.ylabel('Discriminability ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best')
plt.show()

w = np.hstack((eig_pairs[0][1][:,np.newaxis].real,eig_pairs[1][1][:,np.newaxis].real))
np.set_printoptions(suppress=True)
print('Matrix W: \n',w)

X_train_lda = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0]*-1,X_train_lda[y_train==l,1]*-1,c=c,label=l,marker=m)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')
plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr  = LogisticRegression(C=100)
lr = lr.fit(X_train_lda,y_train)
p = plot_decision_regions(X_train_lda,y_train,classifier=lr)
p.title('Training Set')
p.xlabel('LD 1')
p.ylabel('LD 2')
p.legend(loc='best')
p.show()

X_test_lda = lda.transform(X_test_std)
lr.fit(X_test_lda,y_test)
p = plot_decision_regions(X_test_lda,y_test,classifier=lr)
p.title('Test Set')
p.xlabel('LD 1')
p.ylabel('LD 2')
p.legend(loc='best')
p.show()
