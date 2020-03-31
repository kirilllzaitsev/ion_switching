from sklearn import ensemble

MODELS = {
    "RandomForestClassifier": ensemble.RandomForestClassifier(n_estimators=40,n_jobs=8,
                                                              criterion='gini',verbose=2)
}