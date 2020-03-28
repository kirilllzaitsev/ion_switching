import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)