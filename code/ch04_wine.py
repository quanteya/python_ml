import pandas as pd
import numpy as np
from sklearn.datasets import load_wine  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
dt = load_wine()
from ch04_sbs import SBS

df = pd.DataFrame(data=dt.data, columns=dt.feature_names)
df['target'] = pd.Series(dt.target)

X_train,x_test,y_train,y_test = train_test_split(df.iloc[:,:13].values,df.iloc[:,13].values,test_size=0.3,random_state=0)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(x_test)

# lr1 = LogisticRegression(penalty='l1',C=0.1)
# lr2 = LogisticRegression(penalty='l2',C=0.1)
# lr1.fit(X_train_std,y_train)
# lr2.fit(X_train_std,y_train)
# print('\n Training Accuracy LR L1: ',lr1.score(X_train_std,y_train),'   LR L2:',lr2.score(X_train_std,y_train))

# print('\n Testing Accuracy LR L1:  ',lr1.score(X_test_std,y_test),'   LR L2:',lr2.score(X_test_std,y_test))

# fig = plt.figure()
# ax = plt.subplot(111)
# colors = ['blue','green','red','cyan','magenta','yellow','black',
#             'pink','lightgreen','lightblue','gray','indigo','orange']

# weights,params = [],[]

# for c in np.arange(-4.,6.):
#     lr = LogisticRegression(penalty='l1',C=10.**c,random_state=0)
#     lr.fit(X_train_std,y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.**c)

# weights = np.array(weights)
# for column,color in zip(range(weights.shape[1]),colors):
#     plt.plot(params,weights[:,column],label=df.columns[column],color=color)

# plt.axhline(0,color='black',linestyle='--',linewidth=3)
# plt.xlim([10**-5,10**5])
# plt.ylabel('Coefficient')
# plt.xlabel('C')
# plt.xscale('log')
# plt.legend(loc='upper left')
# #ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True )

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel('Accuracy')
plt.xlabel('No of features')
plt.grid()
plt.show()

