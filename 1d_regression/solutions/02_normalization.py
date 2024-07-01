# %%
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_data

###
# A z-value gives the number of standard deviations a data point is from the mean of the data set.
# The mean of the data set (x_1, x_2, ..., x_n) is the sum of the data points divided by the number of data points: 
#   mean = (x_1 + x_2 + ... + x_n) / n.
# The variance of the data set is mean of the squared differences between each data point and the mean: 
#   sd^2 = ((x_1 - mean)^2 + (x_2 - mean)^2 + ... + (x_n - mean)^2) / n.
# The standard deviation sd is the square root of the variance.
# The z-value of a data point x is the difference between x and the mean divided by the standard deviation:
#   z(x) = (x - mean) / sd.
###

x, y = generate_data()

# %%
#TODO: Write a function that calculates the z-values of a data set

def z(x):
    mean = np.mean(x)
    sd = np.std(x)
    return (x - mean) / sd

#TODO: Use the function to calculate the z-values of x and y and plot them

z_x = z(x)
z_y = z(y)

plt.scatter(z_x, z_y, s=1, color='k', alpha=0.5)
plt.show()
# %%

# In many applications, we want to center and scale the data by the _same_ factor.
# This is essentially non-dimensionalization. Removing any extreme numerical factors allows us to build algorithms that are more robust and easier to apply to different data sets.
#TODO: Write a function that centers and scales the data by the same factor, namely the standard deviation of the data set as a whole:
# Build a Nx2 matrix with the first column being the x-values and the second column being the y-values.
# Calculate the column-wise means and the variance of the entire data set.
# The data is given by ((x_1, y_1), (x_2, y_2), ..., (x_n, y_n)). Therefore the mean is a vector as well: (mean_x, mean_y).
# The variance is then:
#   var = ((x_1 - mean_x)^2 + (y_1 - mean_y)^2 + ... + (x_n - mean_x)^2 + (y_n - mean_y)^2) / n.

def non_dimensionalize(x,y):
    xy = np.column_stack((x,y))
    mean = np.mean(xy, axis=0)
    sd = np.std(xy, axis=0)
    sd = (np.sum(sd**2))**0.5
    xy = (xy - mean) / sd
    x, y = xy[:,0], xy[:,1]
    return x, y
