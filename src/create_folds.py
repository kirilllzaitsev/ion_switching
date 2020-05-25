import pandas as pd
from sklearn import model_selection


def main(df):
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False,random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df,y=df.open_channels.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    return df
