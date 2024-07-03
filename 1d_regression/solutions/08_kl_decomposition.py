# %%

import numpy as np
import matplotlib.pyplot as plt
import time

from utils import get_A, get_b, get_coeffs

N = 256
x = np.linspace(0, 1, N)
h = 1/N

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

u_list = []
coeffs_list = []

M = 100

for i in range(M):
    coeffs = get_coeffs()
    kx = k(x)
    A = get_A(N, kx)
    b = get_b(N)
    u = np.linalg.solve(A, b)
    u_list.append(u)
    coeffs_list.append(coeffs)

u_array = np.array(u_list)
coeffs_array = np.array(coeffs_list)

###
# The solutions u(x; k) all look similar. 
# While they have N degrees of freedom, in practice, their collection does not really "use" all that freedom.
# We can try to find a lower-dimensional representation of the solutions using an Eigenvalue Decomposition.
###

###
# The Eigenvectors should live in the same space as the solutions u(x; k). The collection of solutions u(x; k) is stored in a matrix U of size M x N.
# The Eigenvectors should be of size N, hence we are looking to solve
#   U^T U v = lambda v.

evd = np.linalg.eig(u_array.T @ u_array)

lambdas = np.real(evd.eigenvalues)
V = np.real(evd.eigenvectors)


# %%

#TODO: Print the first 5 eigenvalues
print(lambdas[:5])

# %%
#TODO: Plot the first 5 eigenvectors

plt.plot(x, V[:, :5])
plt.show()

# %%

###
# We can now try to express any solution as a linear combination of the eigenvectors. Remember that the original matrix equation was:
#   A u = b
# A linear combination of the eigenvectors is:
#   u = c_1 * v_1 + c_2 * v_2 + ... + c_n * v_n
# When we assemble the eigenvectors into a matrix V, we can write this as:
#  u = V c
# Substituting:
#  A V c = b
# The thing is, now we have N equations for n unknowns (c_1, c_2, ..., c_n). One way to fix this is to demand that the equations are satisfied when we project them onto the eigenvectors:
#  V^T A V c = V^T b
# This is a system of n equations for n unknowns, which we can solve.
###

#TODO: Assemble the matrix V^T A V and vector V^T b using only those Eigenvectors that have non-zero Eigenvalues.
# An Eigenvalue of size < 1e-10 is considered zero.
# Generate random coefficients and solve the system. Time the solve-step.

coeffs = get_coeffs()
kx = k(x)
A = get_A(N, kx)
b = get_b(N)

A_tilde = V[:,:4].T @ A @ V[:,:4]
b_tilde = V[:,:4].T @ b

start_time = time.time()
c = np.linalg.solve(A_tilde, b_tilde)
end_time = time.time()
print(f"Solving took {end_time - start_time} seconds.")

# %%
# TODO: For the same coefficients, solve the full system A u = b. Time it.
start_time = time.time()
u = np.linalg.solve(A, b)
end_time = time.time()
print(f"Solving took {end_time - start_time} seconds.")

# %%
#TODO: Compare the two solutions by plotting them. You can reconstruct the full solution using u_reconstructed = V c, again using only the non-zero Eigenvectors.

u_reconstructed = V[:,:4] @ c

plt.plot(x, u, label="Full solution")
plt.plot(x, u_reconstructed, label="Reconstructed solution")
plt.legend()
plt.show()
# %%

#TODO: Plot the difference between the two solutions.
plt.plot(x, u - u_reconstructed)
# %%
