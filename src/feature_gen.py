import pandas as pd 

TRAIN_DATA = 'input/train_folds.csv'
TEST_DATA = 'input/test_folds.csv'

class Generator():
    def __init__(self, df):
        self.df = df

    def fold_mean(self):
        self.df['fold_mean'] = -1
        for fold in self.df.kfold.unique():
            self.df.loc[self.df['kfold']==fold, 'fold_mean'] = \
                self.df.loc[self.df['kfold']==fold, 'signal'].mean()
        


if __name__ == "__main__":
    # gen = Generator(pd.read_csv(TRAIN_DATA))
    # gen.fold_mean()
    # print(gen.df.columns)
    # gen.df.to_csv('input/train_f.csv',index=False)
    gen = Generator(pd.read_csv(TEST_DATA))
    gen.fold_mean()
    print(gen.df.columns)
    gen.df.to_csv('input/test_f.csv',index=False)