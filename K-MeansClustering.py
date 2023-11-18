#K-Means clustering with Python
import pandas as pd # For reading datasets
import numpy as np # For computations
import matplotlib.pyplot as plt # For visualization
from pandas import DataFrame # For creating data frame
from sklearn.cluster import KMeans
Data={
'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72], 'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
}
df = DataFrame(Data,columns=['x','y'])
# Create and fit the KMeans model
kmeans = KMeans(n_clusters=3).fit(df)
# Find the centroids of the clusters
centroids = kmeans.cluster_centers_
# Get the associated cluster for each data record
kmeans.labels_
# Display the clusters contents and their centroids
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()