from sklearn import ensemble
from xgboost import XGBClassifier

MODELS = {
    "rFOCslow": ensemble.RandomForestClassifier(n_estimators=80,max_depth=2,n_jobs=8,
                                                              class_weight='balanced',verbose=2),
    "rFoth": ensemble.RandomForestClassifier(n_estimators=40,max_leaf_nodes=6,n_jobs=8,
                                                              class_weight='balanced',verbose=2),
    # "rF2": ensemble.RandomForestClassifier(n_estimators=40,max_depth=7,n_jobs=8,
    #                                                               class_weight='balanced',verbose=2),
    # "rF34": ensemble.RandomForestClassifier(n_estimators=40,max_depth=2,n_jobs=8,
    #                                                               class_weight='balanced',verbose=2),
}
