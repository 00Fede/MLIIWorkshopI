import numpy as np
from scipy.linalg import svd

#This function calculates the Eigenvectors corresponding for U and V matrices
class SVD():
    def calcMat(M, opc):
        #Case of V Matrix
        if opc == 1:
            newM = np.dot(M.T, M)
        #Case of U Matrix
        if opc == 2:
            newM = np.dot(M, M.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(newM)
        ncols = np.argsort(eigenvalues)[::-1]

        #Case of V Matrix, let's transpose it
        if opc == 1:
            return eigenvectors[:,ncols].T
        #Case of U, return normally
        else: return eigenvectors[:,ncols]


    #Function that calculates Eigenvalues corresponding to the Sigma Matrix
    def calcD(M):
        if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))):
            newM = np.dot(M.T, M)
        else:
            newM = np.dot(M, M.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(newM)
        eigenvalues = np.sqrt(eigenvalues)
        #Sorting in descending order as the svd function does
        return eigenvalues[::-1]
            