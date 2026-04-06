import numpy as np

def gini(y):
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity

def entropy(y):
    classes = np.unique(y)
    impurity = 0
    for c in classes:
        p = np.sum(y == c) / len(y)
        if p > 0:
            impurity -= p * np.log2(p)
    return impurity