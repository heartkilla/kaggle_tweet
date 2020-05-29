import pandas as pd
from sklearn import model_selection

# 50898
# 666
# 777 
# 21
if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    df = df.dropna().reset_index(drop=True)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=21).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'kfold'] = fold

    df.to_csv("../data/train_folds_new.csv", index=False)
