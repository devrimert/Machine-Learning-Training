import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from pylab import *
import matplotlib.pyplot as plt
# Generate sample data
centers = [[1,1],[-1,-1],[1,-1]]
X, labels_true = make_blobs(n_samples=750,centers=centers,cluster_std=0.4, random_state=0)
# Scale and standardize the dataset
X= StandardScaler().fit_transform(X)
xx, yy = zip(*X)
scatter(xx,yy)
show()
# Set up DBSCAN parameters
db = DBSCAN(eps=0.3,min_samples=10).fit(X)
core_samples = db.core_sample_indices_
core_sample_mask= np.zeros_like(db.labels_,dtype=bool)
core_sample_mask[db.core_sample_indices_] =True
# the number of clusters
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) #if the label is equal to -1, this means the datarecord is an outlier.
# Find the outliers
outliers = X[labels == -1]
# Get the content of each cluster
cluster1 = X[labels == 0]
cluster2 = X[labels == 1]
cluster3 = X[labels == 2]
#Plot the results with a specific color for each cluster and a blac color for noise points
unique_labels = set(labels)
colors = ['y','b','g','r']
for k, col in zip(unique_labels,colors):
    if k==-1:
        col = 'k'
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_sample_mask]
        plt.plot(xy[:,1],'o',markerfacecolor=col,merkeredgecolor='k', markersize=6)
        xy