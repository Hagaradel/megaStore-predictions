from sklearn.preprocessing import LabelEncoder

import numpy as np
import statistics

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        # lbl.fit(list(X[c].values))
        X[c] = lbl.fit_transform(list(X[c].values))
    return X

def scale(X):
    # minmax=list()

    val_min=min(X)
    val_max=max(X)

    Normalized_X=list()
    # X = np.array(X)
    # Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(len(X)):
      Normalized_X.append((X[i] -val_min ) / (val_max- val_min))
    return Normalized_X
def feature_selection(X):
    corr = X.corr()
    top_feature = corr.index[abs(corr['ReturnCategory'])>0.3]
    top_feature = top_feature.delete(-1)
    return top_feature