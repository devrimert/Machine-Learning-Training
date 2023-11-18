# GMM clustering with Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture 
iris = datasets.load_iris() # load the iris dataset 
X = iris.data[:, :2] # select first two columns 
d = pd.DataFrame(X) # turn it into a dataframe 
plt.scatter(d[0], d[1])
plt.show() # plot the data
gmm = GaussianMixture(n_components = 3)
gmm.fit(d) # fit the data as a mixture of 3 Gaussians
labels = gmm.predict(d) # predict the cluster of each data record
print('Converged:',gmm.converged_) # Check if the model has converged 
means = gmm.means_ # get the final “means” for each cluster
covariances = gmm.covariances_ # get the final standard deviations
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
plt.scatter(d0[0], d0[1], c ='red')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='green')
plt.show() # plot the data records in each clusters in different color