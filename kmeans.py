# Applying K-means clustering algorithm to segment customers.  
#
#============================================================
# Author: Marjan Khamesian
# Date: January 2021
#============================================================
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot

import seaborn as sns
import matplotlib.pyplot as plt
# -----------------------------------------------
# Reading data and converting to Pandas DataFrame
customers = pd.read_csv('customers.csv')
print(customers.head())
print("\n================================\n")

print(customers.dtypes)
print("\n================================\n")

# Checking duplication
# ====================
duplication = customers.duplicated().sum()
print(f'Number of duplicated rows: {duplication}')
print("\n================================\n")

# Missing values
# ==============
missing_val = customers.isnull().sum()
print(f'Missing values in each variable: \n{missing_val}')
print("\n================================\n")

# Statistical behaviour
# =====================
print(customers.describe())

# Mean, Standard Deviation, Median, Variance
# ==========================================
def stats(df):
    """ To evaluate some statistics of the dataframe.
    input:
    ======
    df: pd.Dataframe
    return:
    ======
    df_stats: statistic original dataframe
    """
    stats_columns = ['Mean', 'Std', 'Median', 'Variance']
    df_stats = pd.DataFrame(columns=stats_columns)
    for col in df.columns:
        if df[col].dtype == "int64" or df[col].dtype == "float64":
            df_stats.loc[col, 'Mean'] = np.mean(df[col])
            df_stats.loc[col, 'Std'] = np.std(df[col])
            df_stats.loc[col, 'Median'] = np.median(df[col])
            df_stats.loc[col, 'Variance'] = np.var(df[col])
    return df_stats

print(stats(customers))
print("\n================================\n")

# Visualization
# =============

# Scatterplot
customers.plot(kind = 'scatter', x = 'Age', y = 'Spending Score (1-100)') 
pyplot.show()

# Scatterplot and histogram in the same figure
sns.jointplot(x = 'Age', y = 'Spending Score (1-100)', data = customers, height = 5)
pyplot.show()

# Correlation
# =========== 
sns.pairplot(customers, x_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 
               y_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 
               hue = 'Gender', 
               kind = 'scatter',
               palette = 'Set2',
               height = 2,
               plot_kws = {"s": 35, "alpha": 0.8});
pyplot.show()

# Transforming the categorical variable into two binary variables
customers["Male"] = customers.Gender.apply(lambda x: 0 if x == "Male" else 1)
customers["Female"] = customers.Gender.apply(lambda x: 0 if x == "Female" else 1)

X = customers.iloc[:, 2:]
X.head()
print("\n================================\n")

# Dimensionality reduction
# ========================

# Applying PCA and fitting 
# ------------------------
# n_components : number of Principal Components we need to fit our data 
pca = PCA(n_components = 2).fit(X)

# Transformation
pca_2d= pca.fit_transform(X)
print(pca.explained_variance_ratio_)

# Number of components PCA choose after fitting the model 
print('Number of components after fitting is:')
print(pca.n_components_)

# Explained variance : squared-length of the vector
print(pca.explained_variance_)

# ==================
# K-means clustering
# ==================
def euclideandistance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

# WSS : Within-cluster Sum of Square 
# ==================================
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, c = "#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wcss', size=14)
plt.show()

# Kmeans algorithm
# ================
# n_clusters = 5: Number of clusters
# init: k-means++. Smart initialization
# max_iter: Maximum number of iterations of the K-means algorithm for a single run
# n_init: Number of time the algorithm will be run with different centroid seeds 
# random_state: Random number generation for centroid initialization

kmeans = KMeans(n_clusters=5, init = 'k-means++', max_iter = 10, n_init = 10, random_state = 0)

# Fitting & Prediction
# ==================== 
y_means = kmeans.fit_predict(X)

# **************************************
fig, ax = plt.subplots(figsize = (8, 6))

plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y_means, 
            edgecolor="none", 
            cmap=plt.cm.get_cmap("Spectral_r", 5),
            alpha=0.5)

plt.xticks(size=12)
plt.yticks(size=12)

plt.xlabel("Component 1", size = 14, labelpad=10)
plt.ylabel("Component 2", size = 14, labelpad=10)

plt.title('Clusters', size=16)
plt.colorbar(ticks=[0, 1, 2, 3, 4]);
plt.show()

# Centeroid for each cluster
# ==========================
centroid = pd.DataFrame(kmeans.cluster_centers_, columns = ['Age', 'Annual Income', 'Spending Score', 'Male', 'Female'])

# Retrieve the centers
# ====================
centroid.index_name = 'ClusterID'
centroid['ClusterID'] = centroid.index
centroid = centroid.reset_index(drop=True)
print(centroid)

# Prediction for a new customer
# =============================
X_new = np.array([[38, 80, 45, 1, 0]]) 
 
new_customer = kmeans.predict(X_new)
print(f'The cew customer belongs to segment {new_customer[0]}')









