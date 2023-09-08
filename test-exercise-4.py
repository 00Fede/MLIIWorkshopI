np.diag(s).shape
from mymlkit.svd import SVD
def calcMat2(M, opc):
    
    mean = np.mean(M, axis=0)
    X = np.copy(M)
    M = M - mean
    cov = np.cov(M, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # sort the eigenvalues and eigenvectors in decreasing order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # store the first n_components eigenvectors as the principal components
    components = eigenvectors[:, : 20]
    X = X - mean
    X_transformed = np.dot(X, components)
    return X_transformed

def calcMat(M, opc):
    #Case of V Matrix
    if opc == 1:
        #newM = M.T @ M
        newM = np.dot(M.T, M)
    #Case of U Matrix
    if opc == 2:
        #newM = M @ M.T
        newM = np.dot(M, M.T)
    
    #cov = np.cov(newM, rowvar=False)
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
    
    """ if (np.size(M @ M.T) > np.size(M.T @ M)):
        newM = M.T @ M
    else:
        newM = M @ M.T """
    
    eigenvalues, eigenvectors = np.linalg.eig(newM)
    #eigenvalues = np.sqrt(eigenvalues)
    #Sorting in descending order as the svd function does
    return eigenvalues[::-1]


from matplotlib.image import imread
#my_face_file = "averaged_face.jpg"
my_face_file = "Webp-compressed.jpg"
processed_folder_path = "photos_processed"

img = imread(os.path.join(processed_folder_path, my_face_file))

#Calling the corresponding Fuctions and saving the values in variables
Vt = calcMat(img, 1)
U = calcMat(img,2)
Sigma = calcD(img)

#Creating our matrix
A = np.array([[4, 2, 0], [1, 5, 6]])
#Calling the corresponding Fuctions and saving the values in variables
Vt = calcMat(A, 1)
U = calcMat(A,2)
Sigma = calcD(A)

print(Vt,"\n")
print(U, "\n")
print(Sigma)
from matplotlib.image import imread
#my_face_file = "averaged_face.jpg"
my_face_file = "Webp-compressed.jpg"
processed_folder_path = "photos_processed"

img = imread(os.path.join(processed_folder_path, my_face_file))

#Calling the corresponding Fuctions and saving the values in variables
Vt = calcMat(img, 1)
U = calcMat(img,2)
Sigma = calcD(img)


print(Vt,"\n")
print(U, "\n")
print(Sigma)
# plot images with different number of components
comps = [240, 150, 50, 10]
plt.figure(figsize=(12, 6))
  
for i in range(len(comps)):
    low_rank = U[:, :comps[i]] @ np.diag(Sigma[:comps[i]]) @ Vt[:comps[i], :]
      
    if(i == 0):
        plt.subplot(2, 3, i+1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'Actual Image with n_components = {comps[i]}')
      
    else:
        plt.subplot(2, 3, i+1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'n_components = {comps[i]}')