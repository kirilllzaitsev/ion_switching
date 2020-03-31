from sklearn import metrics, preprocessing, ensemble
import pandas as pd
import os 
import joblib
from . import dispatcher
import numpy as np
from tqdm import tqdm 
import gc

TEST_DATA = os.environ.get("TEST_DATA")
# FOLD = int(os.environ.get("FOLD"))
# TEST_DATA = "input/test.csv"
# TRAIN_DATA = "input/train_folds.csv"
# FOLD = 0
MODEL = os.environ.get("MODEL")


def predict():
    print(MODEL)
    df = pd.read_csv(TEST_DATA)

    for FOLD in tqdm(range(5),total=5):
        model = joblib.load(f"models/{MODEL}_{str(FOLD)}.pkl")
        
        if FOLD == 0:
            preds = model.predict(df)
            joblib.dump(preds, f"models/predictions_{str(FOLD)}.pkl")
        else:
            preds += model.predict(df)

    preds = np.true_divide(preds, 5)
    
    preds = pd.DataFrame(np.column_stack((df['time'], preds)), columns=['time', 'open_channels'])
    return preds


if __name__ == "__main__":

    submission = predict()
    submission.open_channels = submission.open_channels.astype(int)
    submission.to_csv("models/submission.csv", float_format="%.4f",index=False)
    # print(metrics.f1_score(y_test, preds, average=None))
    # joblib.dump(clf, f"models/{MODEL}.pkl")