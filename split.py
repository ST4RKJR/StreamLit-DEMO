import numpy as np
from impurity import gini

def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = -1

    parent_impurity = gini(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            left = y[X[:, feature] <= t]
            right = y[X[:, feature] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            weighted_impurity = (
                len(left)/len(y)*gini(left) +
                len(right)/len(y)*gini(right)
            )

            gain = parent_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain