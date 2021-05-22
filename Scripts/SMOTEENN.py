import pandas as pd
import imblearn

from collections import Counter

print("LOADING DATA")
meta_data = pd.read_csv("/mnt/storage/scratch/jc17360/ADS/data/meta_features_train.csv", index_col=0)

y = meta_data["is_fraud"]
X = meta_data.drop(["is_fraud"], axis=1)

print("SAMPLING")
oversample = imblearn.combine.SMOTEENN()

X, y = oversample.fit_resample(X, y)

counter = Counter(y)
print(counter)

X["is_fraud"] = y

X.to_csv("/mnt/storage/scratch/jc17360/ADS/data/SMOTEENN_samples.csv")