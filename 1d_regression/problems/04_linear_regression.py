# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import generate_data, l2_error, z

###
# Linear regression
# We want to find the line y = mx + b that minimizes the sum of the squared errors between the line and the data points:
# One can find an analytical solution and explicit formulae for m and b, or use an optimization algorithm to minimize the error.
###

x, y =  generate_data(unitless=True)

# %%
#TODO: Write a function that takes a vector [m, b] and returns the average of the squared errors between the line y = mx + b and the data points:

# %%
#TODO: Use the scipy.optimize.minimize function to find the values of m and b that minimize the error
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

from scipy.optimize import minimize

#TODO: Plot the data and the line y = m_opt * x + b_opt. Print the values of m_opt, b_opt, and the error

# %%

#TODO: Analytically determine the values of m and b that minimize the error.
# You can assume that the means of both x and y are zero.
# Compare the numerical to the analytical result.

# %%

###
# Lastly, we can generate some test data and see if the model also works on unseen data:
###

x_test, y_test = generate_data(seed=123, unitless=True)


#TODO: In a scatter plot, show the training data, the test data, and the line y = m_opt * x + b_opt.

