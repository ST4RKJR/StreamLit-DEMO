import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig

def plot_split(X, y, feature, threshold):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)

    if feature == 0:
        ax.axvline(x=threshold)
    else:
        ax.axhline(y=threshold)

    return fig

def plot_decision_boundary(X, y, tree, predict_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = [predict_func(tree, point)[0] for point in grid]
    preds = np.array(preds).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y)

    return fig