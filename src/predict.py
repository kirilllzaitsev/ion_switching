from sklearn import metrics, preprocessing, ensemble
import pandas as pd
import os
import joblib
from . import dispatcher
import numpy as np
from tqdm import tqdm
import gc
import pickle
from scipy import stats
import glob

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    print(MODEL)
    test_df = pd.read_csv(TEST_DATA) # len(df) = 2 000 000
    df = test_df.drop(['time','signal'],axis=1)

    batch = 100_000
    chunk = len(df) // batch
    preds = np.zeros((2_000_000,1))

    for FOLD in tqdm(range(5)):
        file = glob.glob(f'models/rF*{FOLD}.pkl')[0]
        with open(file, 'rb') as f:
            model = pickle.load(f)

        if FOLD == 0:  # for batches with sparse open channels < 3
            preds[0*batch:1*batch] = model.predict(df.iloc[0*batch:1*batch]).reshape(-1,1)
            preds[3*batch:4*batch] = model.predict(df.iloc[3*batch:4*batch]).reshape(-1,1)
            preds[8*batch:9*batch] = model.predict(df.iloc[8*batch:9*batch]).reshape(-1,1)
            preds[10*batch:] = model.predict(df.iloc[10*batch:]).reshape(-1,1)

        elif FOLD == 1 or FOLD == 3:  # for 3, 5, 10 channels appeared freq. in a batch
            preds[1*batch:3*batch] += model.predict(df.iloc[1*batch:3*batch]).reshape(-1,1)
            preds[5*batch:8*batch] += model.predict(df.iloc[5*batch:8*batch]).reshape(-1,1)

        elif FOLD == 2 or FOLD == 4:  # most complex batches
            preds[9*batch:10*batch] += model.predict(df.iloc[9*batch:10*batch]).reshape(-1,1)

    with open(f"models/predictions_{str(FOLD)}.pkl", 'wb+') as f:
        pickle.dump(preds, f)


    for i, j in [(1,3),(5,8),(9,10)]:
        preds[i*batch:j*batch] = np.true_divide(preds[i*batch:j*batch], 2)

    print(preds)
    print(stats.describe(preds))
    preds = pd.DataFrame(np.column_stack((test_df['time'], preds)), columns=['time', 'open_channels'])
    return preds


if __name__ == "__main__":

    submission = predict()
    submission.open_channels = submission.open_channels.astype(int)
    submission.to_csv("models/submission.csv", float_format="%.4f",index=False)
