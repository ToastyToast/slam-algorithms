import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def compute_sigma_points(mu, sigma, lambd, alpha, beta):
    n = mu.shape[0]
    sigma_points = np.zeros((n, 2*n+1))

    sigma_points[:, [0]] = mu


    # mroot = np.linalg.cholesky(sigma)
    mroot = sqrtm(sigma)
    mroot = math.sqrt(n+lambd) * mroot

    # compute sigma points
    for i in range(1, n+1):
        # nice indexing
        sigma_points[:,[i]] = mu + mroot[:, [i-1]]

    for i in range(n+1, 2*n+1):
        sigma_points[:,[i]] = mu - mroot[:, [i-n-1]]

    # compute weights for mean and covariance recovery
    w_m = np.zeros(2*n+1)
    w_c = np.zeros(2*n+1)

    w_m[0] = lambd / (n + lambd)
    w_c[0] = w_m[0] + (1 - alpha**2 + beta)
    weight = 1 / ( 2 * (n + lambd))
    for t in range(1,2*n+1):
        w_m[t] = weight
        w_c[t] = weight

    return sigma_points, w_m, w_c


def recover_gaussian(sigma_points, w_m, w_c):
    n = sigma_points.shape[0]
    n_col = sigma_points.shape[1]

    mu = np.zeros((n, 1))

    # Recover mean
    for i in range(0, n_col):
        mu = mu + w_m[i] * sigma_points[:, [i]]

    # Recover covariance
    sigma = np.zeros((n, n))
    for i in range(0, n_col):
        sigma = sigma + w_c[i]*(sigma_points[:, [i]] - mu) * (sigma_points[:, [i]] - mu).T

    return mu, sigma

