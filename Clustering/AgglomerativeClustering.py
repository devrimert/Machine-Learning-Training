# Agglomerative Clustering with Python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Wholesale customers data.csv')
# Normalize the dataset to get all the features at the same scale
from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
# Draw the dendrogram to find the optimum number of clusters
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()
# From the dendrogram, we decide the optimum number of clusters is 2
# apply hierarchical clustering for two clusters only
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)
# visualize the two clusters
plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_) 
plt.show()