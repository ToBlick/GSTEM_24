# %%

import numpy as np
import matplotlib.pyplot as plt
import time

from utils import get_coeffs

###
# We are looking at the displacement of a string made of different materials.
# The string is deformed by its own weight which we assume uniform.
# At both ends, the string is fixed.
# We want to find the temperature distribution along the rod.
# The displacement u(x) at position x satisfies the elasticity equation:
#   d/dx (k(x) du/dx) = 1
# with boundary conditions:
#   u(0) = 0
#   u(1) = 0
###

N = 256
x = np.linspace(0, 1, N)
h = 1/N

###
# The flexibility k(x) is a piecewise constant function.
###
coeffs = get_coeffs()
def k(x):
    if x < 0.25:
        return coeffs[0]
    elif x < 0.5:
        return coeffs[1]
    elif x < 0.75:
        return coeffs[2]
    else:
        return coeffs[3]
k = np.vectorize(k)
kx = k(x)
    
###
# The derivatives are approximated using finite differences:
#   d/dx f(x) = (f(x + h) - f(x)) / h
# h is the spacing between the points in the grid.
# The second derivative is approximated by applying the finite difference formula twice:
#   d^2/dx^2 f(x) = (f(x + h) - 2*f(x) + f(x - h)) / h^2
###

###
# These formulas can be written and computed in matrix form:
#   u = [u_0, u_1, ..., u_N]
# The heat equation is then written as:
#
#   |  h^2     0         0      0  ... 0 |        | u_0 |   |  0 |
#   | k(1)   -2k(1)    k(1)     0  ... 0 |        | u_1 |   |  1 |
#   |   0     k(2)    -2k(2)    0  ... 0 |        | u_2 |   |  1 |
#   |  ...     ...      ...        ...   | /h^2 * | ... | = | ...|    
###

# Let us build the matrix in a loop for clarity:

def get_A(N, k):
    A = np.zeros((N, N))
    A[0, 0] = h**2
    for i in range(1, N-1):
        A[i, i-1] = k(x)[i]
        A[i, i]   = -2 * k(x)[i]
        A[i, i+1] = k(x)[i]
    A[-1, -1] = h**2
    A /= h**2
    return A

# The right-hand side of the equation:
def get_b(N):
    b = np.ones(N)
    b[0] = 0
    b[-1] = 0
    return b

# The solution of this equation is only defined up to a constant, which we set to one:
#A[2, :] = np.ones(N) / N
#b[2] = 1

# Solve the system!
A = get_A(N, k)
b = get_b(N)
start_time = time.time()
u = np.linalg.solve(A, b)
end_time = time.time()
print(f"Solving took {end_time - start_time} seconds.")

# %%

fig, ax1 = plt.subplots()

# Plot on the first y-axis
ax1.plot(x, u, 'k-')
ax1.set_xlabel('x')
ax1.set_ylabel('u', color='k')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(x, k(x), 'c-')
ax2.set_ylabel('k', color='c')

    
# %%

#TODO: Write a loop that solves the equation for 100 different, random k(x) and plots the results.
# Store the solutions in a list u_list.
# Store the coefficients in a list coeffs_list.

u_list = []
coeffs_list = []

for i in range(100):
    coeffs = get_coeffs()
    A = get_A(N, k)
    b = get_b(N)
    u = np.linalg.solve(A, b)
    u_list.append(u)
    coeffs_list.append(coeffs)
# %%
#TODO: Convert the lists to numpy arrays
u_array = np.array(u_list)
coeffs_array = np.array(coeffs_list)

#TODO: Plot the solutions u_list (you can set alpha=0.5 for a neater plot)
plt.plot(x, u_array.T, alpha = 0.5)
#plt.yscale('log')
#plt.ylim(1e-4, 1e1)
plt.show()

# %%
#TODO: Plot a histogram of the average displacements at each point x.
u_average = np.mean(u_array, axis=1)
plt.hist(u_average, bins=25, color='c')
plt.xlabel('u_average')
plt.ylabel('count')
plt.show()

# %%
#TODO: Make a scatterplot of the average displacement versus the average of the coefficients.
# You can use a logarithmic scale.
plt.scatter(np.mean(coeffs_array, axis=1), -u_average, color='c')
plt.xlabel('mean of the coefficients')
plt.ylabel('u_average')
#plt.yscale('log')
#plt.xscale('log')
plt.show()

# %%
