# %%
import numpy as np
import matplotlib.pyplot as plt

###
# Generate some data
###

from utils import generate_data
x, y = generate_data()

###
# Plot the data:
# https://matplotlib.org/stable/gallery/shapes_and_collections/scatter.html
###

#TODO: Make a scatter plot of the data
plt.scatter(x, y, s=1, color='k', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
