from sklearn import metrics, preprocessing, ensemble
import pandas as pd
import os 
import joblib
from . import dispatcher
import numpy as np
from tqdm import tqdm 
import gc

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    print(MODEL)
    df = pd.read_csv(TEST_DATA) # len(df) = 2 000 000
    
    batches = 10
    chunk = len(df) // batches
    preds = []
    for FOLD in tqdm(range(5),total=5):
        model = joblib.load(f"models/{MODEL}_{str(FOLD)}.pkl")
        preds = np.zeros((batches,chunk))
        for batch in range(batches):
            if FOLD == 0:
                preds[batch] = model.predict(df.iloc[chunk*batch:chunk*batch+chunk])
            else:
                preds[batch] += model.predict(df.iloc[chunk*batch:chunk*batch+chunk])

        joblib.dump(preds, f"models/predictions_{str(FOLD)}.pkl")
        del preds

    preds = np.column_stack([joblib.load(f'models/predictions_{str(i)}') for i in range(5)])
    preds = np.true_divide(np.ravel(preds), 5)
    
    preds = pd.DataFrame(np.column_stack((df['time'], preds)), columns=['time', 'open_channels'])
    return preds


if __name__ == "__main__":

    submission = predict()
    submission.open_channels = submission.open_channels.astype(int)
    submission.to_csv("models/submission.csv", float_format="%.4f",index=False)
    # print(metrics.f1_score(y_test, preds, average=None))
    # joblib.dump(clf, f"models/{MODEL}.pkl")