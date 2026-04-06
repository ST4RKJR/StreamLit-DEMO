import numpy as np

def generate_data(n=200, noise=0.1):
    np.random.seed(None)
    X1 = np.random.randn(n//2, 2) + [2, 2]
    X2 = np.random.randn(n//2, 2) + [-2, -2]

    X = np.vstack((X1, X2))
    y = np.array([0]*(n//2) + [1]*(n//2))

    # Add noise
    flip = int(noise * n)
    idx = np.random.choice(n, flip, replace=False)
    y[idx] = 1 - y[idx]

    return X, y