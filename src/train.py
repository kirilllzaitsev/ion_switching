from sklearn import metrics, preprocessing, ensemble
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import os 
import joblib
from . import dispatcher

TRAIN_DATA = os.environ.get("TRAIN_DATA")
FOLD = int(os.environ.get("FOLD"))
# TRAIN_DATA = "input/train_folds.csv"
# FOLD = 0
MODEL = os.environ.get("MODEL")
FOLD_MAPPING = {
	'0': [1,2,3,4],
	'1': [0,2,3,4],
	'2': [0,1,3,4],
	'3': [0,1,2,4],
	'4': [0,1,2,3],
}


if __name__ == "__main__":
    
    threshold = 1
    df = pd.read_csv(TRAIN_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(str(FOLD)))].iloc[:int(len(df)*threshold)]
    val_df = df[df.kfold==FOLD].iloc[:int(len(df)*threshold)]

    y_train = train_df.open_channels.values
    y_test = val_df.open_channels.values

    train_df = train_df.drop("open_channels", axis=1)
    val_df = val_df.drop("open_channels", axis=1)
    train_df = train_df.drop("kfold", axis=1)
    val_df = val_df.drop("kfold", axis=1)

    # val_df = val_df[train_df.columns]

    #n_jobs = -1 may lead to errors!
    clf = dispatcher.MODELS.get(MODEL)
    # clf = ensemble.RandomForestClassifier(n_estimators=32, n_jobs=4,
	# criterion='gini',verbose=2)
    clf.fit(train_df, y_train)

    preds = clf.predict(val_df)
    # f1_scorer = make_scorer(f1_score, average='macro')

    print(metrics.f1_score(y_test, preds, average='macro'))
    joblib.dump(clf, f"models/{MODEL}_{str(FOLD)}.pkl")