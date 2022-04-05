import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import plotnine as p9
from sklearn.naive_bayes import GaussianNB, CategoricalNB
plt.style.use('ggplot')  # set the theme for plots
mpl.rcParams['figure.figsize'] = (10, 8)

import warnings

warnings.filterwarnings('ignore')

# https://anson.ucdavis.edu/~jsharpna/DSBook/unit1/cost_of_uni.html
# vars of interest:

"""
INPUT:
    controllable:
        PREDDEG - Predominant undergraduate degree awarded
                    0 Not classified
                    1 Predominantly certificate-degree granting
                    2 Predominantly associate's-degree granting
                    3 Predominantly bachelor's-degree granting
                    4 Entirely graduate-degree granting
        HIGHDEG - highest degree awarded
        ADM_RATE - admission rate
        SAT_AVG - avg SAT score
        ACTCMMID - Midpoint of the ACT cumulative score
        TUITIONFEE_IN - in state tuition
        TUITIONFEE_OUT - oos tuition
        UGDS - Enrollment of undergraduate certificate/degree-seeking students
        UGDS_WHITE / UGDS_BLACK / UGDS_HISP / UGDS_ASIAN / UGDS_AIAN / UGDS_NHPI / UGDS_NHPI / UGDS_2MOR / UGDS_NRA
            / UGDS_UNKN /  UGDS_WHITENH / UGDS_BLACKNH / UGDS_API / UGDS_AIANOLD / UGDS_HISPOLD - diversity percentages
        AVGFACSAL - avg faculty salary
        UG25ABV -Percentage of undergraduates aged 25 and above
        UGDS_MEN - share of enrollment men
        UGDS_WOMEN
        GRADS - grad students #
        PRGMOFR - Number of programs offered
        BOOKSUPPLY - Cost of attendance: estimated books and supplies
        ROOMBOARD_ON- Cost of attendance: on-campus room and board
        
        
    uncontrollable:
        UNITID - id of inst
        INSTNM - name
        CONTROL - 1 public, 2 private nonprofit, 3 private for profit
        PCTFLOAN - Percent of all undergraduate students receiving a federal student loan
        ICLEVEL - inst level :   1=4year  2=2-year 3=less than 2 year
        CCBASIC - carnegie classification
        
OUTPUT:
        C150_4 - Completion rate for first-time, full-time students at four-year institutions 
                (150% of expected time to completion)
        C150_4_POOLED - Completion rate for first-time, full-time students at less-than-four-year institutions 
                        (150% of expected time to completion), pooled for two year rolling averages
        CDR2 - Two year cohort default rate
        CDR3 - Three year cohort default rate
        DEATH_YR2_RT - % died within 2 years at original institution
        COMP_ORIG_YR2_RT - Percent completed within 2 years at original institution
        COMP_4YR_TRANS_YR2_RT - Percent who transferred to a 4-year institution and completed within 2 years
        WDRAW_ORIG_YR2_RT - Percent withdrawn from original institution within 2 years
        COMPL_RPY_3YR_RT - Three-year repayment rate for completers
        MDEARN_PD - Overall Median earnings of students working and not enrolled 10 years after entry
        MN_EARN_WNE_P10 - Mean earnings of students working and not enrolled 10 years after entry
        OVERALL_YR4_N - Number of students in overall 4-year completion cohort
            low medium high # of students in 4 year completion
            LO_INC_YR4_N 
            MD_INC_YR4_N
            HI_INC_YR4_N

        RET_FT4 - retention rate of full time students

NEED TO FIND EARNINGS
"""
vars_interest = ['ADM_RATE', 'UGDS', 'TUITIONFEE_IN', 'TUITIONFEE_OUT', 'MN_EARN_WNE_P10', 'PREDDEG',
                 'HIGHDEG', 'ADM_RATE', 'SAT_AVG', "ACTCMMID", "AVGFACSAL", "UG25ABV", "UGDS_WOMEN", "UGDS_MEN"
                "GRADS", "BOOKSUPPLY", "ROOMBOARD_ON",

                  "UNITID", "CONTROL", "PCTFLOAN", "ICLEVEL"]

output_vars = ["C150_4", "RET_FT4", "CDR2", "DEATH_YR2_RT", "COMP_ORIG_YR2_RT", "COMP_4YR_TRANS_YR2_RT",
               "WDRAW_ORIG_YR2_RT", "COMPL_RPY_3YR_RT", "MD_EARN_WNE_P10", "OVERALL_YR4_N"
               ]
# Which dataumns have no NAs
datadir = 'CollegeScorecard_Raw_Data_03142022'
data = pd.read_csv(datadir + '/MERGED2009_10_PP.csv')

# Which dataumns have no NAs
# data_dna = data.dropna(axis=1)
data_dtypes = dict(data.dtypes.replace([np.dtype('int64'), np.dtype('float64')]))  # make the dtypes floats
data_dtypes['UNITID'] = np.dtype('int64')  # convert the UNITID back to int
data_dtypes['INSTNM'] = np.dtype('str')
data_dtypes.update({a: np.dtype('float64') for a in vars_interest})  # make them floats
data_dtypes.update({a: np.dtype('float64') for a in output_vars})  # make them floats
"""
data_try_again = pd.read_csv(datadir + '/MERGED2009_10_PP.csv', na_values='PrivacySuppressed',
                             dtype=data_dtypes)
data_try_again.info()
data_try_again['Year'] = pd.Period('2010', freq='Y')
"""

def read_cs_data(year, data_dtypes, datadir):
    """read a datalegeScorecard dataframe"""
    nextyr = str(int(year) + 1)[-2:]
    filename = datadir + '/MERGED{}_{}_PP.csv'.format(year, nextyr)
    data = pd.read_csv(filename, na_values='PrivacySuppressed',
                       dtype=data_dtypes)
    data['Year'] = pd.Period(str(int(year) + 1), freq='Y')
    return data


data = pd.concat((read_cs_data(str(y), data_dtypes, datadir) for y in range(2010, 2019)))
data = data.set_index(['UNITID', 'Year'])
data.to_csv('everything.csv')

"""
data.head()
dataClean = data[data['UGDS'] > 1000]
duq = dataClean.query('CITY=="Pittsburgh" and STABBR=="PA" and INSTNM=="Duquesne University"')
duq = duq.reset_index(level=0)
duq['YearDT'] = duq.index.to_timestamp()

dataClean = dataClean.reset_index(level=1)
dataClean['YearDT'] = pd.PeriodIndex(dataClean['Year']).to_timestamp()
# if you want some columns add them here.
final_table_columns = ['INSTNM', 'Year']
final_table_columns.extend(output_vars)
final_table_columns.extend(vars_interest)
dataClean = dataClean[dataClean.columns.intersection(final_table_columns)]
dataClean.dropna()
print(dataClean.columns)
"""
y = '2019'
print("""
Duq Uni Statistics {}
Admissions rate:{}, Undergrad admissions:{:.0f},
In-state tuition: {:.0f}, Out-of-state tuition: {:.0f},
4 year completion cohort: {:.0f},
Women: {},
SAT AVG: {:.0f}
      """.format(y, *tuple(duq.loc[y, ['ADM_RATE', 'UGDS', 'TUITIONFEE_IN',
                                       'TUITIONFEE_OUT', 'OVERALL_YR4_N', "UGDS_WOMEN", "SAT_AVG"]])))

# dataClean = dataClean[dataClean['PREDDEG'] == 3]
#
# print(dataClean.info())
# print(dataClean.head())
#
# dataClean.to_csv('dataframe.csv')

"""

X = dataClean.drop(columns="OVERALL_YR4_N")
y = dataClean["OVERALL_YR4_N"]


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape, X_test.shape)

import category_encoders as ce
encoder = ce.OneHotEncoder(cols=X.columns)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
print(X_train.head())
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB

# instantiate the model
nb = GaussianNB()
#nb = CategoricalNB()
# fit the model
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = nb.predict(X_train)

y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(nb.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(nb.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
"""





