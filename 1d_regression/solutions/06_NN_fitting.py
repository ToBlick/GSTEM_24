# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import generate_data, l2_error, z

###
# Neural Net regression
# Next, we replace the polynomial with a neural network: y = NN(x, theta).
# theta are the parameters of the NN. They play the role that a played for the polynomial model.
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
###

x, y = generate_data(N=10_000, unitless=True)
x_train, y_train = x.reshape(-1, 1), y.reshape(-1, 1)

from sklearn.neural_network import MLPRegressor

MLP = MLPRegressor(hidden_layer_sizes=(10,10,10),       # Number of neurons in the hidden layer(s)
                   activation='tanh',               # Activation function: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                   solver='adam',                   # {‘lbfgs’, ‘sgd’, ‘adam’}
                   alpha=0.0001,                    # L2 regularization term
                   batch_size=512,                  # batch size, defaults to 200 or data set size, whichever is smaller
                   learning_rate_init=0.001, 
                   max_iter=200, 
                   shuffle=True,                    # Shuffle samples in each iteration
                   random_state=None,               # Random seed for the stochastic optimizer
                   tol=0.0001,                      # stopping tolerance (relative)
                   verbose=True, 
                   beta_1=0.9,                      # adam optimizer parameters
                   beta_2=0.999, 
                   epsilon=1e-08, 
                   n_iter_no_change=10)             # Number of iterations with no improvement to wait before stopping

MLP.fit(x_train, y_train)

# %%

#TODO: Plot the data and the neural network predictions.

x_test, y_test = generate_data(N=1_000, unitless=True, seed=123)
x_test, y_test = x_test.reshape(-1, 1), y_test.reshape(-1, 1)

_x = np.linspace(-4, 4, 100).reshape(-1, 1)
plt.scatter(x_test, y_test, s=1, c='k')
plt.scatter(x_test, MLP.predict(x_test), s=1, c='c')
# %%
