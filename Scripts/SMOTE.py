import pandas as pd
import imblearn

print("LOADING DATA")
meta_data = pd.read_csv("data/meta_features_train.csv", index_col=0)

y = meta_data["is_fraud"]
X = meta_data.drop(["is_fraud"], axis=1)

print("SAMPLING")
oversample = imblearn.over_sampling.SMOTE(sampling_strategy=1.0)

X, y = oversample.fit_resample(X, y)

X["is_fraud"] = y

X.to_csv("data/SMOTE_samples.csv")