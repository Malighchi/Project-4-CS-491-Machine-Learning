import pca
import numpy as np
import compress

# X = np.array([[1, 1], [1, 0], [2, 2], [2, 1], [2, 4], [3, 4], [
#              3, 3], [3, 2], [4, 4], [4, 5], [5, 5], [5, 7], [5, 4]])
# Z = pca.compute_Z(X, True, True)
# COV = pca.compute_covariance_matrix(Z)
# # print(COV)
# L, PCS = pca.find_pcs(COV)
# Zstar = pca.project_data(Z, PCS, L, 1, 0)
# print(PCS)
# print(Zstar)
X = compress.load_data('Data/Train/')
compress.compress_images(X, 10)
compress.compress_images(X, 100)
compress.compress_images(X, 500)
compress.compress_images(X, 1000)
compress.compress_images(X, 2000)
