from sklearn import ensemble, svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

MODELS = {
    "rFOCslow": ensemble.RandomForestClassifier(n_estimators=80,max_depth=2,n_jobs=8,
                                                              verbose=2),
    "rF13": ensemble.RandomForestClassifier(n_estimators=40,max_leaf_nodes=8,n_jobs=8,
                                                              verbose=2),
    "rF24": ensemble.RandomForestClassifier(n_estimators=40,max_depth=8,n_jobs=8,class_weight='balanced',
                                                              verbose=2),
    "lr24": LogisticRegression(max_iter=200, class_weight='balanced',n_jobs=8,solver='lbfgs'),
    "xgb24": XGBClassifier(n_estimators=80,max_depth=10,n_jobs=8,
                                                              verbose=2),
}
