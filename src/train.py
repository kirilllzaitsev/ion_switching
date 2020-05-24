from sklearn import metrics, preprocessing, ensemble
from sklearn.model_selection import learning_curve, StratifiedKFold, ShuffleSplit
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np
import os
import joblib
import pickle
import matplotlib.pyplot as plt
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

def single_train_score_lc(estimator, train_df, target, train_sizes, cv):
    f1_scorer = make_scorer(f1_score, average='macro')

    train_sizes, train_scores, validation_scores = learning_curve(
            estimator, train_df, target,
            cv = cv, scoring = f1_scorer)

    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,40)
    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    f1_scorer = make_scorer(f1_score, average='macro')
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=f1_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


if __name__ == "__main__":

    threshold = 1
    df = pd.read_csv(TRAIN_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(str(FOLD)))].iloc[:int(len(df)*threshold)]
    val_df = df[df.kfold==FOLD].iloc[:int(len(df)*threshold)]

    y_train = train_df.open_channels.values
    y_test = val_df.open_channels.values

    train_df = train_df.drop(["open_channels","kfold"], axis=1)
    val_df = val_df.drop(["open_channels", "kfold"], axis=1)

    print(train_df.columns)
    clf = dispatcher.MODELS.get(MODEL)

    # plt.figure(figsize = (16,5))
    # single_train_score_lc(clf, train_df, y_train, train_sizes=1, cv=2)

	fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    title = "Learning Curves (RF)"
    Cross validation with 100 iterations to get smoother mean test and train
    score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

    plot_learning_curve(clf, title, train_df, y_train, axes=axes[:, 0],ylim=(0.7, 1.01),
                        cv=cv, n_jobs=1)

    plt.show()

    clf.fit(train_df, y_train)
	
    del train_df, y_train

    preds = clf.predict(val_df)



    print(metrics.f1_score(y_test, preds, average='macro'))

    with open(f"models/{MODEL}_{str(FOLD)}.pkl", "wb+") as f:
        pickle.dump(clf, f)
