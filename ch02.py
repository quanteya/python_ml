import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class MyPerceptron(object): 
    def __init__(self, eta=0.01,n_iter=10,shuffle=True, random_seed=1):
        self.eta = eta 
        self.n_iter=n_iter
        self.shuffle = shuffle

        if random_seed: 
            np.random.seed(random_seed)
    
    def fit(self,X,y):

        self._initialize_weights(X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors=0

            if self.shuffle:
                X,y = self._shuffle(X,y)

            for xi,target in zip(X,y):
                update = self.eta*(target- self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+= int(update!=0)
            self.errors_.append(errors)
        return self

    def _shuffle(self,X,y):
        r = np.random.permutation(len(y))
        return X[r],y[r]
    
    def _initialize_weights(self, m):
        self.w_ = np.random.normal(loc=0,scale=1,size=1+m)
        self.w_initialized = True
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X)>=0,1,-1)    


class AdalineGD(MyPerceptron): 

    def __init__(self, eta=0.01,n_iter=10,shuffle=False,random_seed=1):
        super().__init__(eta=eta,n_iter=n_iter,shuffle=shuffle,random_seed=random_seed)
    
    def fit(self,X,y):

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self,X,y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi,target)
        else: 
            self._update_weights(X,y)

        return self




    def activation(self,X):
        return self.net_input(X)

    def _update_weights(self,xi,target):
        output = self.net_input(xi)
        errors = target - output
        self.w_[1:] += self.eta*xi.T.dot(errors)
        self.w_[0] += self.eta*errors.sum()
        cost = (errors**2).sum()/2
        return cost 
            
        

    def predict(self,X):
        return np.where(self.activation(X)>=0.0,1,-1)    


def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers=('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #decision surface 
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha=0.8,c=cmap(idx),
                    edgecolors='black',marker=markers[idx],
                    label=cl)

    if test_idx:
        X_test,y_test = X[test_idx,:] , y[test_idx]
        
        plt.scatter(X_test[:,0],X_test[:,1],c='',
                    alpha=1.0,
                    edgecolors='black',
                    linewidths=1,marker='o',s=55,
                    label='test set')

    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend(loc='upper left')



    return plt
"""
add_fig should be false first time and later set to true to keep adding

"""
def plt_dec_regions(X,y,classifier,test_idx=None,resolution=0.02,ax=None):
    markers=('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])


    if ax is None:
        ax = plt.gca()

    #decision surface 
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    

    #plot class samples
    for idx,cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha=0.8,c=cmap(idx),
                    edgecolors='black',marker=markers[idx],
                    label=cl)

    if test_idx:
        X_test,y_test = X[test_idx,:] , y[test_idx]
        
        ax.scatter(X_test[:,0],X_test[:,1],c='',
                    alpha=1.0,
                    edgecolors='black',
                    linewidths=1,marker='o',s=55,
                    label='test set')





    return ax


def plots(X,y,colorMap=False):
    
    if colorMap:
        print('Ets do something')

    else:
        plt.plot(X,y,marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of misclassifications')

    plt.show()

def graph_scatter(X,y):
    plt.scatter(X[0:50,0],X[0:50,1],marker='o',c='red',label='setosa')
    plt.scatter(X[50:,0],X[50:,1],marker='x',c='blue',label='versicolor')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.legend(loc='upper left')
    plt.show()



