#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from ch02 import Perceptron,plots,graph_scatter,plot_decision_regions,AdalineGD


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data',header=None)

y = df.iloc[0:100,4].values 
X = df.iloc[0:100,[0,2]].values
y = np.where(y=='Iris-setosa',-1,1)

"""
#pp = Perceptron(eta=0.1,n_iter=10,shuffle=True)
#pp.fit(X,y)

#plots(range(1,len(pp.errors_)+1),pp.errors_)

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ada1 = AdalineGD(eta=0.001,n_iter=10)
ada1.fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaline - Learning rate '+str(ada1.eta))
ada2 = AdalineGD(eta=0.0001,n_iter=10)
ada2.fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(sum-squared-error)')
ax[1].set_title('Adaline - Learning rate '+str(ada2.eta))

plt.show()
"""

X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

ada = AdalineGD(eta=0.01,n_iter=15,shuffle=False,random_seed=11)
ada.fit(X_std,y)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
p = plot_decision_regions(X_std,y,classifier=ada)
p.xlabel('Sepal length [standardized]')
p.ylabel('Petal length [standardized]')
p.title('Adaline - Gradient Descent')

ax[1] = p
ax[0].plot(range(1,len(ada.cost_)+1),(ada.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaline - Learning rate '+str(ada.eta))

plt.show()

