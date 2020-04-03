
from sklearn import model_selection
import pandas as pd

TRAIN_DATA = 'input/train.csv'

class CrossValidation:
    def __init__(self, df, target_cols, 
    problem_type='multiclass_classification',n_folds=5,
    shuffle='False',random_state=42):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.rs = random_state
        self.problem_type = problem_type
        self.mlbl_delimiter = None

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            target = self.target_cols[0]
            unique_vals = self.df[target].nunique()
            if unique_vals == 1:
                raise Exception("Only one unique value found!")
            
            elif unique_vals > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.n_folds, 
                                                    shuffle=False)
                
                for fold, (_, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                    self.df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):

            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            kf = model_selection.KFold(n_splits=self.n_folds)
            for fold, (_, val_idx) in enumerate(kf.split(X=self.df)):
                self.df.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.df) * holdout_percentage / 100)
            self.df.loc[:len(self.df) - num_holdout_samples, "kfold"] = 0
            self.df.loc[len(self.df) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":

            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.mlbl_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)


        else:
            raise Exception("Problem type not understood!")

        
        self.df.kfold = self.df.kfold.astype('int')
        return self.df

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_DATA)
    cv = CrossValidation(df, shuffle=True, target_cols=["open_channels"], 
                         problem_type="multiclass_classification")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())