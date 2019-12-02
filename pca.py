import numpy as np
import math
from numpy import linalg as LA


def compute_Z(X, centering=True, scaling=False):
    X_copy = np.copy(X.astype(float))
    sampleSize = np.size(X_copy, 0)
    featureSize = np.size(X_copy, 1)
    if(centering):
        for col in range(featureSize):
            mean = np.mean(X_copy[:, col])
            for row in range(sampleSize):
                X_copy[row][col] = X_copy[row][col] - mean
    if(scaling):
        for col in range(featureSize):
            std = np.std(X_copy[:, col], ddof=1)
            for row in range(sampleSize):
                X_copy[row][col] = X_copy[row][col] / std
    return X_copy


def compute_covariance_matrix(Z):
    return np.matmul(np.transpose(Z), Z)


def find_pcs(COV):
    eigenValues, eigenVectors = LA.eig(COV)

    eigenValues = eigenValues.real
    eigenVectors = eigenVectors.real

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    return eigenValues, eigenVectors


def project_data(Z, PCS, L, k, var):
    necc_EVs = k
    if(var != 0):
        sum = np.sum(L)
        numer = 0.0
        for i in range(np.size(L, 0)):
            numer += L[i]
            if((numer/sum) >= var):
                necc_EVs = i
                break
    return np.matmul(Z, PCS[:, 0:necc_EVs])
