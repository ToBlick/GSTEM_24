import numpy as np
import jax.numpy as jnp
from jax import random

def generate_data(N=1000, epsilon=20, seed=42, unitless=False):
    np.random.seed(seed)
    x = 10 * np.random.randn(N) + 7
    y = 3*x + 4 * x * np.sin(x/10) + epsilon * np.random.randn(N)
    if unitless:
        x, y = non_dimensionalize(x, y)
    return x, y

def generate_data_jax(key, N=1000, epsilon=10):
    x_key, xi_key = random.split(key)
    x = 10 * random.normal(x_key, (N,)) + 7
    y = 3*x + 4 * x * jnp.sin(x/10) + epsilon * random.normal(xi_key, (N,))
    return x / 100, y / 200

def l2_error(x, y, m, b):
    return np.sum((y - (m*x + b))**2) / len(y)

def z(x):
    mean = np.mean(x)
    sd = np.std(x)
    return (x - mean) / sd

def non_dimensionalize(x,y):
    xy = np.column_stack((x,y))
    mean = np.mean(xy, axis=0)
    sd = np.std(xy, axis=0)
    sd = (np.sum(sd**2))**0.5
    xy = (xy - mean) / sd
    x, y = xy[:,0], xy[:,1]
    return x, y

def get_A(N, kx):
    h = 1/N
    A = np.zeros((N, N))
    A[0, 0] = h**2
    for i in range(1, N-1):
        A[i, i-1] = kx[i]
        A[i, i]   = -2 * kx[i]
        A[i, i+1] = kx[i]
    A[-1, -1] = h**2
    A /= h**2
    return A

def get_b(N):
    b = np.ones(N)
    b[0] = 0
    b[-1] = 0
    return b

def get_coeffs():
    return 10 ** (np.random.rand(4) * 4 - 2)