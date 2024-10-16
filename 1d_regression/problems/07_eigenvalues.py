# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import generate_data, l2_error, z

###
# We can treat the data as pairs (x_i, y_i). The simplest predictor for any y_i is the mean.
# The error of the mean predictor is the variance of the data.
# A better predictor is to add the information of the x_i, and fit a linear model.
# A direction in 2D is a normed vector. The dot product of the direction and the data gives the projection of the data on that direction.
###

x, y = generate_data(unitless=True)

#TODO: Build a data matrix of size N x 2, where the first column is x and the second column is y. 

data = np.column_stack((x, y))

#TODO: generate a random direction vector of size 2 and plot it together with the data.
# Hint: generate random angles in [0, 2*pi] and use cos and sin to get the direction.
# You can use plt.quiver to plot the vector.

alpha = np.random.uniform() * 2 * np.pi
direction = np.array([np.cos(alpha), np.sin(alpha)])

plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.quiver(0, 0, direction[0], direction[1], scale=5, color='c')

# %%

#TODO: For the random direction, calculate the projection of the data on the direction.
# The projection of a vector v on a direction d is the dot product of v and d times d
# Recall that the data is Nx2 and one direction is 2x1.
# The result should again be a Nx2 array.

data_projected = np.outer(np.dot(data, direction), direction)

# %%
#TODO: Calculate the difference between the data and the projection for each direction.

data_residuals = data - data_projected

#TODO: Calculate the mean square error of the projection for each direction.

error = np.mean(np.sum(data_residuals**2, axis=1))

# %%

#TODO: Plot the data, the projection, and the projection error. 
# Repeat this for a few random or hand-picked directions. Which direction gives the smallest error?

plt.scatter(x, y, s=1, color='k', alpha=0.5)
alpha = np.random.uniform() * 2 * np.pi
direction = np.array([np.cos(alpha), np.sin(alpha)])
plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.quiver(0, 0, direction[0], direction[1], scale=5, color='c')
data_projected = np.outer(np.dot(data, direction), direction)
data_residuals = data - data_projected
error = np.mean(np.sum(data_residuals**2, axis=1))
plt.title(f'error = {error:.3e}')
# %%

###
# The direction that minimizes the error is the eigenvector of the covariance matrix of the data.
# The covariance matrix is a d x d matrix, where d is the number of dimensions of the data.
# Since the data is centered, the covariance matrix is the dot product of the data matrix with its transpose.
###

#TODO: Calculate the covariance matrix of the data and its eigenvectors and eigenvalues. Print them.
# Hint: Matrix multiplication is done with @ in numpy.
# Hint: The np.linalg.eig function returns the eigenvalues and eigenvectors of a matrix.

cov = data.T @ data
eigenvalues, eigenvectors = np.linalg.eig(cov)
print(eigenvalues)
print(eigenvectors)

# %%

#TODO: Calculate the error of the projection on the eigenvectors.

plt.scatter(x, y, s=1, color='k', alpha=0.5)
direction = eigenvectors[1]
plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.quiver(0, 0, direction[0], direction[1], scale=5, color='c')
data_projected = np.outer(np.dot(data, direction), direction)
data_residuals = data - data_projected
error = np.mean(np.sum(data_residuals**2, axis=1))
plt.title(f'error = {error:.3e}')
# %%
