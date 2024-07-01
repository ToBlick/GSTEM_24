# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import generate_data, l2_error, z

from ipywidgets import FloatSlider, interactive, VBox, Output
from IPython.display import display

###
# Linear regression by hand
###

x, y = generate_data(unitless=True)
    
m0 = 0
b0 = 0
# Create ipywidgets sliders 
m_slider = FloatSlider(value=m0, min=-9, max=9, step=0.01, description='m')
b_slider = FloatSlider(value=b0, min=-3, max=3, step=0.01, description='b')

# Output widget to display the plot
out = Output()

# Create the initial plot to get the limits
fig, ax = plt.subplots()
ax.scatter(x, y, s=1, color='k', alpha=0.5)
line, = ax.plot(x, m0 * x + b0, color='c')
error = l2_error(x, y, m0, b0)
ax.set_title(f'm = {m0:4.2f}, b = {b0:4.2f}, error = {error:4.2f}')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.close(fig)

def update(m, b):
    with out:
        out.clear_output(wait=True)
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1, color='k', alpha=0.5)
        line, = ax.plot(x, m * x + b, color='c')
        error = l2_error(x, y, m, b)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'm = {m:4.2f}, b = {b:4.2f}, error = {error:4.4f}')
        plt.show()

interactive_plot = interactive(update, m=m_slider, b=b_slider)
display(VBox([m_slider, b_slider, out]))

# display
update(m0, b0)

#TODO: Determine (roughly) the values of m and b that minimize the error by hand