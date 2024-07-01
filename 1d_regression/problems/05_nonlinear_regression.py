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
#TODO: Write a function to minimize the mismatch between the data and the curve to optain the optimal coefficients a_0, a_1, a_2, a_3

#TODO: Plot the data and the curve. Print the the error and compare to the linear fit.

# %%

#TODO: Do the same for a polynomial of arbitrary degree. What happens when the degree is very large (e.g. > 50)?
