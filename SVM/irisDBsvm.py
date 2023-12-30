import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))
#['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']


print(iris.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2
# 3                4.6               3.1                1.5               0.2
# 4                5.0               3.6                1.4               0.2


df['target'] = iris.target # Added target column to the data set.
print(df.head())
#   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
#0                5.1               3.5                1.4               0.2       0
#1                4.9               3.0                1.4               0.2       0
#2                4.7               3.2                1.3               0.2       0
#3                4.6               3.1                1.5               0.2       0
#4                5.0               3.6                1.4               0.2       0

print(iris.target_names)
#['setosa' 'versicolor' 'virginica']

print(df[df.target ==1].head()) #Show the first 5 element that target is virginica.
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
#50                7.0               3.2                4.7               1.4       1
#51                6.4               3.2                4.5               1.5       1
#52                6.9               3.1                4.9               1.5       1
#53                5.5               2.3                4.0               1.3       1
#54                6.5               2.8                4.6               1.5       1


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())
#   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
#0                5.1               3.5                1.4               0.2       0      setosa
#1                4.9               3.0                1.4               0.2       0      setosa
#2                4.7               3.2                1.3               0.2       0      setosa
#3                4.6               3.1                1.5               0.2       0      setosa
#4                5.0               3.6                1.4               0.2       0      setosa


from matplotlib import pyplot as plt
# seperate dataframe according to target.
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='red', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='+')

#plt.show()

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='red', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='+')

#plt.show()

from sklearn.model_selection import train_test_split #train ve test icin verileri verilen yuzdeye gore ayirmak icin kullanacagiz.

X = df.drop(['target','flower_name'], axis='columns')
print(X.head())
#   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
#0                5.1               3.5                1.4               0.2
#1                4.9               3.0                1.4               0.2
#2                4.7               3.2                1.3               0.2
#3                4.6               3.1                1.5               0.2
#4                5.0               3.6                1.4               0.2

y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2) # %20 lik dilimi test icin ayiriyoruz. 150 veriden 120 si train icin 30 u ise test icin.
print(len(X))
#150 total
print(len(X_train))
#120 train
print(len(X_test))
#30 test

from sklearn.svm import SVC
# model = SVC()
# model.fit(X_train, y_train) #svc modeli train verilerini kullanarak egitiyoruz

# model_score = model.score(X_test, y_test)
# print(model_score)
#0.9666666666666667 => yani sadece bir veri yanlis gruplanmis. Modelin skoru %96.6

#biz svc fonksiyonunu kullandigimizda herhangi bir parametre ozellestirmesi yapmadigimiz icin regularization degeri (C) 1.0 olarak alinmisti.
#simdi onu 10 a yukseltiyorum.

model10 = SVC(C=10)
model10.fit(X_train,y_train)
model10_score = model10.score(X_test,y_test)
print(model10_score)

print("-------Done---------")