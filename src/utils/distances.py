import numpy as np

def euclidean_dist(x, y):
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    if d != y.shape[1]:
        raise Exception
    x = np.repeat(x[:,None,:], repeats=m, axis=1)
    y = np.repeat(y[None,:,:], repeats=n, axis=0)
    return ((x-y)**2).sum(2)


def cosine_dist(x, y):
    x_norm = (x.T / np.linalg.norm(x, axis=1)).T
    y_norm = (y.T / np.linalg.norm(y, axis=1)).T
    return 1 - x_norm @ y_norm.T


def calc_dist(x, y, dist: str="cos"):
    if dist.upper() in ["EUC", "EUCLIDEAN"]:
        return euclidean_dist(x, y)
    elif dist.upper() in ["COS", "COSINE"]:
        return cosine_dist(x, y)
    else:
        raise ValueError