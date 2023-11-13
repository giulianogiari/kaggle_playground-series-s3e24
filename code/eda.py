
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

import seaborn as sns
import matplotlib.pyplot as plt

import shap
from sklearn.inspection import permutation_importance

#
# define constant variables
#
INPUT_PATH = '/Users/giulianogiari/Desktop/kaggle/playground-series-s3e24/data/'
FIG_PATH = '/Users/giulianogiari/Desktop/kaggle/playground-series-s3e24/figures/'
RANDOM_STATE = 0

# read the data
train = pd.read_csv(INPUT_PATH + 'train.csv')

# get the target variable and remove it from the training set
y = train['smoking']
X = train.drop(['smoking', 'id'], axis=1)

#
# check the data
#
X.head()

# .t allows to see all the columns in the database
X.describe().T

#
# data visualization
#
# correlation matrix
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(X.corr(), cmap='RdBu_r', vmax=1, vmin=-1, center=0, 
            square=True, linewidths=.5, cbar_kws={'shrink':.5, 'label':'Correlation Coefficient'}, 
            annot=True, fmt='.1f', ax=ax)
#plt.show()
plt.savefig(FIG_PATH + 'correlation_matrix.png', dpi=150)
plt.close()

# pairplot
#sns.pairplot(data = train, hue = 'smoking', corner = True)

# explore the target variable
sns.countplot(x=y)

#
# define a pipeline for classification
#

# we will use a gradient boosting classifier as implemented in sklearn
# and we will use cross validation to get an estimate of the performance
# of the model

clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

cv = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)

#
# train and text the model on the training data
#

results = cross_validate(clf, X, y, cv=cv, scoring='roc_auc', return_estimator=True)

# get the mean and standard deviation of the cross validation scores
print(f"ROC AUC: {results['test_score'].mean():.4f} +/- {results['test_score'].std():.4f}")

# 
# use shap to look af feature importance
#

shap_values = shap.TreeExplainer(results['estimator'][0]).shap_values(X)

N_X = 4
N_Y = -(-len(X.keys()) // N_X)

fig, ax = plt.subplots(N_Y, N_X, figsize=(20, 14))
ax = ax.flatten()

for i, f in enumerate(X.keys()):
    # show interactions with respect to age
    # interaction_index = 7
    # if f == 'age':
    #     interaction_index = 'auto'
    shap.dependence_plot(
        f, shap_values, X,
        # interaction_index=7, 
        # use automatic "most" important interaction
        ax=ax[i], show=False)

# remove left-over facets
for i in range(len(X.keys()), N_X * N_Y):
    ax[i].remove()
    
fig.suptitle('SHAP, Partial Dependance', 
        x=0, horizontalalignment='left', fontsize=25)
fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.savefig(FIG_PATH + 'shap_dependence_plots.png', dpi=150)

#
# feature importance
#

fi_cv = pd.DataFrame()
for est in results['estimator']:
    imp = permutation_importance(est, X, y, scoring='roc_auc', random_state=RANDOM_STATE)
    fi_cv = pd.concat([fi_cv, pd.DataFrame({'Importance': imp.importances_mean,
                                            'Feature': X.columns})])

# plot feature importance
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=fi_cv, ax=ax, errorbar='sd')
sns.swarmplot(x='Importance', y='Feature', data=fi_cv, ax=ax, size=4, color='white', 
              edgecolor='k', linewidth=1)
plt.show()
plt.savefig(FIG_PATH + 'permutation_feature_importance.png', dpi=150)

# average values of feature importance and 
# print the features above 75 percentile of importance
fi_avg = fi_cv.groupby('Feature').mean().sort_values(by='Importance', ascending=False)
features = fi_avg[fi_avg['Importance'] > fi_avg['Importance'].quantile(.25)].index

# 
# use mutual information to select features
# https://www.kaggle.com/code/ryanholbrook/mutual-information
#
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y, discrete_features=X.dtypes == int)
mi_scores = pd.DataFrame({'MI_Scores': mi_scores, 'Feature': X.columns})
mi_scores = mi_scores.sort_values(by='MI_Scores', ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x='MI_Scores', y='Feature', data=mi_scores, ax=ax)
plt.savefig(FIG_PATH + 'mutual_information.png', dpi=150)
plt.close()

#features =['height(cm)',  'Gtp',  'hemoglobin',  'triglyceride',  'age',  'LDL']
from sklearn.model_selection import cross_val_score
results = cross_val_score(clf, X[features], y, cv=cv, scoring='roc_auc')
# print results
print(f"ROC AUC: {results.mean():.4f} +/- {results.std():.4f}")

#
# feature engineering
#
X['feat1']=X['systolic']/X['relaxation']

import xgboost as xgb
# Use "hist" for constructing the trees, with early stopping enabled.
clf = xgb.XGBClassifier(tree_method="hist")
results = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
# print results
print(f"ROC AUC: {results.mean():.4f} +/- {results.std():.4f}")

