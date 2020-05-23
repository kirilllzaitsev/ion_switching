from sklearn import ensemble
from xgboost import XGBClassifier

MODELS = {
    "rF": ensemble.RandomForestClassifier(n_estimators=40,n_jobs=8,
                                                              class_weight='balanced_subsample',verbose=2),
    "extraT": ensemble.ExtraTreesClassifier(n_estimators=40,n_jobs=8,
                                                            class_weight='balanced_subsample',verbose=2)
}
