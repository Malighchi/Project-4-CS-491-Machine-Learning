import pca
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(input_dir):
    # This just gets data from the files and flattens them
    data = np.array([[plt.imread(root+filename).flatten() for filename in files]
                     for root, dirs, files in os.walk(top=input_dir)], dtype=np.int32)

    print(data[0].shape)

    # The data is a 3D thing, but I just want the first thing
    return np.transpose(data[0].astype(float))


def compress_images(DATA, k):
    # Do PCA
    Z = pca.compute_Z(DATA, centering=True, scaling=True)

    COV = pca.compute_covariance_matrix(np.transpose(Z))
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(np.transpose(Z), PCS, L, k, 0)

    # Find X compressed
    U = np.transpose(PCS[:, 0:k])
    X_compressed = np.matmul(Z_star, U)

    # Images have values from 0 to 255, so rescale
    for row, image in enumerate(X_compressed):
        min = np.amin(image)
        max = np.amax(image)
        for col, pixel in enumerate(image):
            X_compressed[row][col] = (
                X_compressed[row][col] - min) * (255/(max - min))

    X_compressed = X_compressed.astype(int)

    # Output everything into output dir
    if not os.path.exists('Output'):
        os.mkdir('Output')
    imageNumber = 0

    for image in X_compressed:
        filename = 'Output/k' + str(k) + 'image' + str(imageNumber) + '.png'
        # Love hard coded numbers <3
        plt.imsave(filename, image.reshape(60, 48), cmap='gray')
        imageNumber += 1

    return
