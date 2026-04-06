import numpy as np
from split import best_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=3, min_samples=5):
    if depth >= max_depth or len(set(y)) == 1 or len(y) < min_samples:
        return Node(value=np.bincount(y).argmax())

    feature, threshold, gain = best_split(X, y)

    if feature is None:
        return Node(value=np.bincount(y).argmax())

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth, min_samples)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth, min_samples)

    return Node(feature, threshold, left, right)