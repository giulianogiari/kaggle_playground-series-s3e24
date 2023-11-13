
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

#
# define constant variables
#
INPUT_PATH = '/Users/giulianogiari/Desktop/kaggle/playground-series-s3e24/data/'
FIG_PATH = '/Users/giulianogiari/Desktop/kaggle/playground-series-s3e24/figures/'
SUB_PATH = '/Users/giulianogiari/Desktop/kaggle/playground-series-s3e24/submission/'
RANDOM_STATE = 0

# read the data
train = pd.read_csv(INPUT_PATH + 'train.csv')

# get the target variable and remove it from the training set
y = train['smoking']
X = train.drop(['smoking', 'id'], axis=1)

# we will use a gradient boosting classifier as implemented in sklearn
# and we will use cross validation to get an estimate of the performance
# of the model

clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

# train the classifier on the training set
clf.fit(X, y)

# get the predictions on the test set
test = pd.read_csv(INPUT_PATH + 'test.csv')
X_test = test.drop('id', axis=1)
# predict the probability and retain the second column, 
# i.e.the  probability of the positive class
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
y_pred = clf.predict_proba(X_test)[:,1]

# create a submission file
submission = pd.read_csv(SUB_PATH + 'sample_submission.csv')
assert(all(test['id'].values==submission['id'].values))
submission['smoking'] = y_pred
submission.to_csv(SUB_PATH + 'submission.csv', index=False)
