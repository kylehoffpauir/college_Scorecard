import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import tensorflow as tf
import plotnine as p9
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

# calculate similarity between institutions with bayes:
similar = ['PREDDEG', 'HIGHDEG', 'ADM_RATE', 'SAT_AVG', "ACTCMMID", 'TUITIONFEE_IN', 'TUITIONFEE_OUT',
           'UGDS',"AVGFACSAL","UGDS_WOMEN", "UGDS_MEN", "BOOKSUPPLY", "ROOMBOARD_ON", "CONTROL",
           "PCTFLOAN", "INEXPFTE", "C150_4"]
# do naive bayes on X for output y for each y in similar
# take top 5 features that impact the goodness

data = pd.read_csv('cleanedData.csv')
print(data.info())
print(data.head())
X =  data[similar]
encoder = OneHotEncoder(categories='auto')
X = encoder.fit_transform(data)
y = data['UNITID']

#print(X.query('INSTNM=="Duquesne University"'))

"""
    Metrics explained:

    Homogeniety: similarity of cluster's elements

    Completeness : same label as cluster

    V-measure : mean of homogeniety and completeness

    Silhouette score : how similar an object is to its own cluster .
    
"""
def k_means(true_k):
    homo, comp, v, sil = 0, 0, 0, 0
    num_iter = 5
    for i in range(num_iter):
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
        model.fit(X)
        labels = model.labels_
        homo += metrics.homogeneity_score(X, labels)
        comp += metrics.completeness_score(X, labels)
        v += metrics.v_measure_score(X, labels)
        sil += metrics.silhouette_score(X, labels, sample_size=1000)
    print("METRICS FOR k = %i" % true_k)
    print("Homogeneity: %0.3f" % (homo / num_iter))
    print("Completeness: %0.3f" % (comp / num_iter))
    print("V-measure: %0.3f" % (v / num_iter))
    print("Silhouette Coefficient: %0.3f" % (sil / num_iter))


# sil = []
# K = range(2, 25)
# for k in K:
#     km = KMeans(n_clusters=k, max_iter=200, n_init=10)
#     km.fit(X)
#     labels = km.labels_
#     sil.append(metrics.silhouette_score(X, labels, metric = 'euclidean'))
#     print(k)
#
# plt.plot(K, sil, 'bx-')
# plt.xlabel('k')
# plt.ylabel('silhouette score')
# plt.title('Silhouette Method For Optimal k')
# plt.show()
#7 or 10

#k_means(15, False)
#k_means(10)
#k_means(25, False)
#k_means(35, False)

model = KMeans(n_clusters=10, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
pred = model.predict(X)
frame = pd.DataFrame(X)
data['cluster'] = pd.Series(pred, index=data.index)
print(model.cluster_centers_)

print(data.info())
duq = data.loc[data['INSTNM'] == "Duquesne University"]


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
     print(duq)

duqClust = duq['cluster'].value_counts().index.tolist()[0]

clusterOfInterest = data.loc[data['cluster'] == duqClust]
clusterOfInterest.to_csv('clusterData.csv')
