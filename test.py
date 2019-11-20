import pca 
import numpy as np 
X = np.array([[1, 1], [1, 0] , [2, 2], [2, 1], [2, 4], [3, 4], [3, 3], [3, 2], [4, 4], [4, 5], [5, 5], [5, 7], [5, 4]])
Z = pca.compute_Z(X, True, True)
#print(Z)
COV = pca.compute_covariance_matrix(Z)
L , PCS = pca.find_pcs(COV)
Zstar = pca.project_data(Z,PCS ,L ,1 ,0)
#print(L) 
# print(Zstar)