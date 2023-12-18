import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/hp/Desktop/IRIS.CSV')
df
df.head()
df.tail()
sns.catplot(x = 'species', hue = 'species', kind = 'count', data = df)
plt.show()
plt.bar(df['species'],df['petal_width'])
plt.show()
sns.set()
sns.pairplot(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']])
plt.show()
df.describe()
df.columns
df.info()
df
X = df.drop(['species'], axis=1)
X
Label_Encode = LabelEncoder()
Y = df['species']
Y = Label_Encode.fit_transform(Y)
Y
df['species'].nunique()
df['species'].nunique()
X
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
X_train
X_train.shape
X_test.shape
Y_test.shape
Y_train.shape
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(X_train)
X_train_std = standard_scaler.transform(X_train)
X_test_std = standard_scaler.transform(X_test)
X_train_std
Y_train
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std,Y_train)
predict_knn=knn.predict(X_test_std)
accuracy_knn=accuracy_score(Y_test,predict_knn)*100
accuracy_knn
df
color_map=np.array(['Red','green','blue'])
figure=plt.scatter(df['petal_length'],df['petal_width'],c=color_map[Y],s=30)
X
