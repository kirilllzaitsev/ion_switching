from sklearn import ensemble
from xgboost import XGBClassifier

MODELS = {
    "rF": ensemble.RandomForestClassifier(n_estimators=80,n_jobs=8,
                                                              criterion='gini',verbose=2),
    "extraT": ensemble.ExtraTreesClassifier(n_estimators=80,n_jobs=8,
                                                            criterion='gini',verbose=2)                                                  
}