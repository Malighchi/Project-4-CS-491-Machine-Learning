import numpy as np
import math
from numpy import linalg as LA


def compute_Z(X, centering=True, scaling=False):
    X_copy = np.copy(X.astype(float))
    sampleSize = np.size(X_copy, 0)
    featureSize = np.size(X_copy, 1)
    if(centering):
        for i in range(featureSize):
            mean = np.mean(X_copy[:, i])
            for j in range(sampleSize):
                X_copy[j][i] = X_copy[j][i] - mean
    if(scaling):
        for i in range(featureSize):
            std = np.std(X_copy[:, i])
            for j in range(sampleSize):
                X_copy[j][i] = X_copy[j][i] / std
    return X_copy


def compute_covariance_matrix(Z):
    return np.matmul(np.transpose(Z), Z)


def find_pcs(COV):
    eigenValues, eigenVectors = LA.eig(COV)
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

    # print(np.transpose(Z))
    # print(Z)
    # print(PCS)
    # print(PCS[0:necc_EVs, :])
    # print(PCS)
    return np.matmul(Z, np.transpose(PCS[0:necc_EVs, :]))
    # return 'peanutes'
