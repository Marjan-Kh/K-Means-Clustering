# This program walks through clustering  
# by applying Kmeans to a random datasets.  

#========================================
# Author: Marjan Khamesian
# Date: January 2021
#========================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate two different random datasets 
X1 = -2 * np.random.rand(50,2)
X2 = 1 + 2 * np.random.rand(50,2)

# Concatenation
X = np.r_[X1,X2]

# Scatter plot
plt.scatter(X[ : , 0], X[ :, 1], s = None, c = 'g')
plt.show()

# Apply Kmeans 
# ============
from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=2) # n_cluster: an arbitrary value of 2 
print(kmean.fit(X))

# Find the centroid 
# =================
centroid = kmean.cluster_centers_
print(f'The center of the clusters locate at {centroid}')

# Plot the centroids 
# ==================
plt.scatter(X[ : , 0], X[ : , 1], s =None, c='g')
plt.scatter(2.04179805,  2.09814033, s=100, c='b', marker='o')
plt.scatter(-0.94538668, -1.03439097, s=100, c='r', marker='o')
plt.show()

# Test
kmean.labels_

# Prediction for a new random numbers
# ===================================
X_new = np.array([[2.5,3]]) 
new_sample = kmean.predict(X_new)
print(f'The new numbers belong to cluster {new_sample[0]}')









    




