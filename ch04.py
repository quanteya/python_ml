import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder

df = pd.read_csv('Book1.csv',header=0,sep='\t')

size_mapping = {'XL':3 ,'L':2, 'M':1 }

inv_size_mapping = {v:k for k,v in size_mapping.items()}

df['size']=df['size'].map(size_mapping)
# imr = Imputer()
# imp_df = imr.fit(df)
# imputed_data = imp_df.transform(df.values)
print (df)
le = LabelEncoder()
y = le.fit_transform(df['classlabel'].values)

X = df[['color','size','price']].values
col_le = LabelEncoder()
X[:,0] = col_le.fit_transform(X[:,0])
one_le = OneHotEncoder(categorical_features=[0])
X =one_le.fit_transform(X).toarray()


print(X,y)