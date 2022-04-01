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
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # set the theme for plots
mpl.rcParams['figure.figsize'] = (10, 8)
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

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

# vars_interest = ['ADM_RATE', 'UGDS', 'TUITIONFEE_IN', 'TUITIONFEE_OUT',  'PREDDEG',
#                                    'HIGHDEG', 'ADM_RATE', 'SAT_AVG', "ACTCMMID", "DEBT_MDN", "AVGFACSAL", "UGDS_WOMEN",
#                                    "UGDS_MEN", "GRADS", "BOOKSUPPLY", "ROOMBOARD_ON", "NUM4_PRIV",  "INEXPFTE",
#                                    "UNITID", "CONTROL", "PCTFLOAN", "ICLEVEL"]
vars_interest = ['ADM_RATE', 'UGDS', 'TUITIONFEE_IN', 'TUITIONFEE_OUT',  'PREDDEG',
                 'HIGHDEG', 'ADM_RATE', 'SAT_AVG', "ACTCMMID", "DEBT_MDN", "AVGFACSAL", "UGDS_WOMEN",
                 "UGDS_MEN", "GRADS", "BOOKSUPPLY", "ROOMBOARD_ON", "NUM4_PRIV",  "INEXPFTE",
                 "UNITID", "CONTROL", "PCTFLOAN", "ICLEVEL",
                 "PCIP01", "PCIP03", "PCIP04", "PCIP05", "PCIP09", "PCIP10", "PCIP11", "PCIP12", "PCIP13",
                 "PCIP14", "PCIP15", "PCIP16", "PCIP19", "PCIP22", "PCIP23", "PCIP24", "PCIP25", "PCIP26",
                 "PCIP27", "PCIP29", "PCIP30", "PCIP31", "PCIP38", "PCIP39", "PCIP40", "PCIP41", "PCIP42",
                 "PCIP43", "PCIP44", "PCIP45", "PCIP46", "PCIP47", "PCIP48", "PCIP49", "PCIP50", "PCIP51",
                 "PCIP52", "PCIP54"]

output_vars = ["C150_4", "RET_FT4", "CDR2", "COMP_ORIG_YR2_RT",
               "COMPL_RPY_3YR_RT", "OVERALL_YR4_N"]


data = pd.read_csv('clusterData.csv')
print(data.info())
print(data.head())

data.dropna(how='any', inplace=True)
X = data[vars_interest]
encoder = OneHotEncoder(categories='auto')
X = encoder.fit_transform(data)
#y = data["C150_4"]

dict = {}
for o in output_vars:
    y = data[o]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    train_score_list = []
    test_score_list = []
    # tested using every solver available and got the same
    # weighted this way bc id rather mistakenly id a poison than an an edible
    # more cautious
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    y_pred_test = reg.predict(x_test)
    y_pred_val = reg.predict(x_val)
    coef_dict = {}
    for coef, feat in zip(reg.coef_,vars_interest):
        coef_dict[feat] = coef
    dict[o] = coef_dict
    print(reg.score(X,y))

# now we have regressed over every output and stored the coeffs with their associated value
# we need to go through each coef and normalize their values
#       tot = sum(abs(coef))
#       normalizedC = c/total --> this gives us a -1 to 1 value for importance
#       now go through before we are finished with our current output value and check the top 10 best and worst coefs
#       only store those in our dictionaries.
# then we can make new duquense test data using our best and worst features
# minimize and maximize them to find the values for optimal duquense


print(dict)

# print(reg.score(x_test, y_test))
# print("training accuracy: " + str(accuracy_score(y_train, y_pred_train)))
# print("testing accuracy: " + str(accuracy_score(y_test, y_pred_test)))
# print("validation accuracy: " + str(accuracy_score(y_val, y_pred_val)))
#
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
# print(classification_report(y_test, y_pred))
# print(reg.get_params())
# from sklearn.feature_selection import RFE
# data_final_vars=data.columns.values.tolist()
#
# linreg = LinearRegression()
# rfe = RFE(linreg)
# rfe = rfe.fit(X, y)
# print(rfe.support_)
# print(rfe.ranking_)
# import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary2())