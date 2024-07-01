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

def error(params):
    m, b = params
    return np.mean((y - (m*x + b))**2)

# %%
#TODO: Use the scipy.optimize.minimize function to find the values of m and b that minimize the error
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

from scipy.optimize import minimize

minimizer = minimize(error, [0, 0])
m_opt, b_opt = minimizer.x

#TODO: Plot the data and the line y = m_opt * x + b_opt. Print the values of m_opt, b_opt, and the error

print(f'm_opt = {m_opt:.3e}, b_opt = {b_opt:.3e}, error = {error([m_opt, b_opt]):.3e}')

plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.plot(x, m_opt*x + b_opt, color='c')
plt.show()

# %%

#TODO: Compare to the analytical result

m_analytical = np.mean(x*y) / np.mean(x**2)
b_analytical = np.mean(y) - m_analytical * np.mean(x)

print(f"m_analytical = {m_analytical:.3e}, b_analytical = {b_analytical:.3e}, error = {error([m_analytical, b_analytical]):.3e}")
# %%

###
# Lastly, we can generate some test data and see if the model also works on unseen data:
###

x_test, y_test = generate_data(seed=123, unitless=True)

def error(params, x, y):
    m, b = params
    return np.mean((y - (m*x + b))**2)

print(f"error on test data = {error([m_opt, b_opt], x_test, y_test):.3e}")

plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.scatter(x_test, y_test, s=1, color='r', alpha=0.5)
plt.scatter(x_test, m_opt*x_test + b_opt, s=1, color='c', alpha=0.5)
plt.show()

# %%
