# Applying K-means clustering algorithm to segment customers.  

#============================================================
# Author: Marjan Khamesian
# Date: January 2021
#============================================================
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------
# Reading data and converting to Pandas DataFrame
customers = pd.read_csv('customers.csv')
customers.head()
customers.dtypes

# Checking duplication
duplication = customers.duplicated().sum()
print(f'Number of duplicated rows: {duplication}')

# Missing values
missing_val = customers.isnull().sum()
print(f'Missing values in each variable: \n{missing_val}')

# Statistical behaviour
def stats(variable):
    df_stats = pd.DataFrame([variable.name, np.mean(variable), np.std(variable), np.median(variable), np.var(variable)],
                            ['Variable', 'Mean', 'Standard Deviation', 'Median', 'Variance']).set_index('Variable')

    if variable.dtype == 'int64' or variable.dtype == 'float64':
        return df_stats
    else:
        return pd.DataFrame(variable.value_counts())
    
 









