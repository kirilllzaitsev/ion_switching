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

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    print(MODEL)
    df = pd.read_csv(TEST_DATA) # len(df) = 2 000 000
    
    batch = 400_000
    chunk = len(df) // batch
    #preds = np.zeros((400_000,1))
    for FOLD in tqdm(range(5)):

        with open(f"models/{MODEL}_{str(FOLD)}.pkl", 'rb') as f:
            model = pickle.load(f)

        preds = np.zeros((batch,1))
        
        #if FOLD == 0:
        preds = model.predict(df.iloc[FOLD*batch:FOLD*batch+batch])
            #preds = model.predict(df)
        # else:
        #     preds += model.predict(df.iloc[FOLD*batch:FOLD*batch+batch])

        with open(f"models/predictions_{str(FOLD)}.pkl", 'wb+') as f:
            pickle.dump(preds, f)
        
    preds = np.hstack([joblib.load(f'models/predictions_{str(i)}.pkl') for i in range(5)])
    #preds = np.true_divide(preds, 5).reshape((2_000_000,5))
    #preds = np.sum(preds, axis=1)
    print(preds)
    print(stats.describe(preds))
    preds = pd.DataFrame(np.column_stack((df['time'], preds)), columns=['time', 'open_channels'])
    return preds


if __name__ == "__main__":

    submission = predict()
    submission.open_channels = submission.open_channels.astype(int)
    submission.to_csv("models/submission.csv", float_format="%.4f",index=False)
    # print(metrics.f1_score(y_test, preds, average=None))
    # joblib.dump(clf, f"models/{MODEL}.pkl")