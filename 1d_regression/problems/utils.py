import numpy as np

def generate_data(N=1000, epsilon=20, seed=42, unitless=False):
    np.random.seed(seed)
    x = 10 * np.random.randn(N) + 7
    y = 3*x + 4 * x * np.sin(x/10) + epsilon * np.random.randn(N)
    if unitless:
        x, y = non_dimensionalize(x, y)
    return x, y

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