'''
http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
short version with sklearn PCA

'''
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler  # normalizacion
from sklearn.decomposition import PCA

df = pd.read_csv(
    #filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    filepath_or_buffer='../iris.data.txt',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

# split data table into data X and class labels y
X = df.ix[:,0:4].values
y = df.ix[:,4].values

# normalize X
X_std = StandardScaler().fit_transform(X)

sklearn_pca = PCA(n_components=2) # ? why 2
Y_sklearn = sklearn_pca.fit_transform(X_std)



plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(X_std[y==lab, 2],
                X_std[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.legend(loc='lower center')
plt.tight_layout()


plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                    ('blue', 'red', 'green')):
    plt.scatter(Y_sklearn[y==lab, 0],
                Y_sklearn[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.legend(loc='lower center')
plt.tight_layout()
'''
'''
plt.show()