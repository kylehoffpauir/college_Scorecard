import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from minisom import MiniSom
from sklearn.cluster import KMeans
import tensorflow as tf
import plotnine as p9
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


plt.style.use('ggplot')  # set the theme for plots
mpl.rcParams['figure.figsize'] = (10, 8)
from sklearn.datasets import make_regression

import warnings
warnings.filterwarnings('ignore')

# https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
# https://rubikscode.net/2021/07/06/implementing-self-organizing-maps-with-python-and-tensorflow/
# https://towardsdatascience.com/self-organizing-map-layer-in-tensroflow-with-interactive-code-manual-back-prop-with-tf-580e0b60a1cc
# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
data = pd.read_csv('dataframe.csv')
print(data.info())
print(data.head())


"""
USING MIMISOM LIBRARY
som = MiniSom(6, 6, 4, sigma=0.5, learning_rate=0.5)
som.train_random(data, 100)
"""

from sklearn.model_selection import train_test_split
X = data
encoder = OneHotEncoder(categories='auto')
X = encoder.fit_transform(X)
y = data.columns

# ok so what do we do?

# prato statistics will allow to optimize for two things @ once, maybe not useful now
#   multivariate analysis of variance (MANOVA) is a procedure for comparing
#   multivariate sample means. As a multivariate procedure, it is used when
#   there are two or more dependent variables, and is often followed by significance
#   tests involving individual dependent variables separately.

# classify?
# naive bayes would let us see the feature importance of features
# kmeans seems to be a problem
# PCA would let us see a graph but is that even useful
# KNN maybe?
# i just want to see what duquesne is close to.

# regression analysis for the improvements
    # take the closest 100 schools over all years to duquesne
    #


# calculate distortion for a range of number of cluster
"""
distortions = []
for i in range(1, 20):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

km = KMeans(
    n_clusters=15, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(X)
print(y_km)
"""

"""
    Homogeniety: degree to which clusters contain element of the same class

    Completeness : degree to which all elements belonging to certain category are found in a cluster

    V-measure : mean of homogeniety and completeness

    Silhouette score : how similar an object is to its own cluster .
    

print("Homogeneity: %0.3f" % metrics.homogeneity_score(X.target, y))
print("Completeness: %0.3f" % metrics.completeness_score(X.target, y))
print("V-measure: %0.3f" % metrics.v_measure_score(X.target, y))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, y, sample_size=1000))
      """