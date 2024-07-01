# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import generate_data, l2_error, z

###
# Polynomial regression
# Instead of fitting a line, we want to fit a polynomial of degree n to the data:
# y = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_0
###

x, y = generate_data(unitless=True)

# %%
#TODO: Write a function that takes a vector a and x and returns the curve y = a_3 x^3 + a_2 x^2 + a_1 x + a_0
#TODO: Write a function that returns the average of the squared errors between the curve and the data points gven a and x

def cubic_curve(a, x):
    return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3

def cubic_fit(a):
    return np.mean((y - cubic_curve(a, x))**2)

from scipy.optimize import minimize

minimizer = minimize(cubic_fit, np.zeros(4))
a_opt = minimizer.x

#TODO: Plot the data and the curve. Print the the error and compare to the linear fit.

print(f'error = {cubic_fit(a_opt):.3e}')

plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.scatter(x, cubic_curve(a_opt, x), s=1, color='c', alpha=0.5)
plt.show()
# %%

#TODO: Do the same for polynomial of different degrees

degree = 10

def poly_curve(a, x):
    return np.sum([a[i]*x**i for i in range(len(a))], axis=0)

def poly_fit(a):
    return np.mean((y - poly_curve(a, x))**2)

a0 = np.zeros(degree+1)

minimizer = minimize(poly_fit, a0)
a = minimizer.x

print(f'error = {poly_fit(a):.3e}')

plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.scatter(x, poly_curve(a_opt, x), s=1, color='c', alpha=0.5)
plt.show()

# %%
