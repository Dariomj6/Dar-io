# -*- coding: utf-8 -*-
"""
Created on Wed May 04 17:55:03 2016

@author: Ricardo y Dario
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

#in x1,x2,x3,x4 we obtain the mean form of each column 


a= X.mean(0)
print a

"""adding all the columns to form a matrix, and transposing the matrix"""
DataSN = X-a
DataSNtrans = np.transpose(DataSN)

Cmatrix = np.dot(DataSNtrans, DataSN) #Obtain the covariance matrix 


w, v = LA.eig(Cmatrix) #Obtains both the eigenvalues and eigenvectors

"""delete the smallest of both the values and the vectors"""
w1 = np.delete(w, 3)
v1 = np.delete(v, 3, 1)

"""Obtain the transpose from the vectors and from x"""
v_trans = np.transpose(v1)
x_trans = np.transpose(X)

"""Obtain the new data set"""
NewDS = np.dot(v_trans, x_trans)
n_trans = np.transpose(NewDS)

"""Ploting the 3 D graph"""
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(n_trans[y == label, 0].mean(),
              n_trans[y == label, 1].mean() + 1.5,
              n_trans[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(n_trans[:, 0], n_trans[:, 1], n_trans[:, 2], c=y, cmap=plt.cm.spectral)

x_surf = [n_trans[:, 0].min(), n_trans[:, 0].max(),
          n_trans[:, 0].min(), n_trans[:, 0].max()]
y_surf = [n_trans[:, 0].max(), n_trans[:, 0].max(),
          n_trans[:, 0].min(), n_trans[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()