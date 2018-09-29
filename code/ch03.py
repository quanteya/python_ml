import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ch02 import plot_decision_regions,plt_dec_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target
sc = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(3, 3)

X_combined = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

pp = Perceptron(max_iter=40,eta0=0.1,random_state=0)
#pp = SGDClassifier(loss='perceptron')
pp.fit(X_train_std,y_train)

ax = plt.subplot(gs[0,0])
fig = plt_dec_regions(X_combined,y_combined,classifier=pp,test_idx=range(105,150))
y_pred = pp.predict(X_test_std)
plt.title('Perceptron Accuracy: '+str(accuracy_score(y_test,y_pred)))

lr = LogisticRegression(C=1000,random_state=0)
#lr = SGDClassifier(loss='log')
lr.fit(X_train_std, y_train)

ax = plt.subplot(gs[0,1])
fig = plt_dec_regions(X_combined,y_combined,classifier=lr,test_idx=range(105,150))
y_pred = lr.predict(X_test_std)
plt.title('Logistic Regression Accuracy: '+str(accuracy_score(y_test,y_pred)))

svm = SVC(C=1.0,kernel='linear',random_state=0)
#svm = SGDClassifier(loss='hinge')
svm.fit(X_train_std,y_train)

ax = plt.subplot(gs[0,2])
fig = plt_dec_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
y_pred = svm.predict(X_test_std)
plt.title('SVM Accuracy: '+str(accuracy_score(y_test,y_pred)))

svm = SVC(C=1.0,kernel='rbf',random_state=0,gamma=0.2)
#svm = SGDClassifier(loss='hinge')
svm.fit(X_train_std,y_train)

ax = plt.subplot(gs[1,0])
fig = plt_dec_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
y_pred = svm.predict(X_test_std)
plt.title('SVM RBF Gamma:0.2  Accuracy: '+str(accuracy_score(y_test,y_pred)))

svm = SVC(C=1.0,kernel='rbf',random_state=0,gamma=10)
#svm = SGDClassifier(loss='hinge')
svm.fit(X_train_std,y_train)

ax = plt.subplot(gs[1,1])
fig = plt_dec_regions(X_combined,y_combined,classifier=svm,test_idx=range(105,150))
y_pred = svm.predict(X_test_std)
plt.title('SVM RBF Gamma:10  Accuracy: '+str(accuracy_score(y_test,y_pred)))


dtr = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
dtr.fit(X_train_std,y_train)

ax = plt.subplot(gs[1,2])
fig = plt_dec_regions(X_combined,y_combined,classifier=dtr,test_idx=range(105,150))
y_pred = dtr.predict(X_test_std)
plt.title('Decision Tree  Accuracy: '+str(accuracy_score(y_test,y_pred)))
export_graphviz(dtr,out_file='entropy_decision_tree.dot',feature_names=['petal length','petal width'])


rdtr = RandomForestClassifier(criterion='entropy',n_jobs=2,n_estimators=10,random_state=1)
rdtr.fit(X_train_std,y_train)

ax = plt.subplot(gs[2,0])
fig = plt_dec_regions(X_combined,y_combined,classifier=rdtr,test_idx=range(105,150))
y_pred = rdtr.predict(X_test_std)
plt.title('Decision Tree  Accuracy: '+str(accuracy_score(y_test,y_pred)))
#export_graphviz(rdtr,out_file='random_forest_decision_tree.dot',feature_names=['petal length','petal width'])


knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)
y_pred = knn.predict(X_test_std)

ax = plt.subplot(gs[2,1])
fig = plt_dec_regions(X_combined,y_combined,classifier=knn,test_idx=range(105,150))
y_pred = knn.predict(X_test_std)
plt.title('KNN  Accuracy: '+str(accuracy_score(y_test,y_pred)))

plt.show()
