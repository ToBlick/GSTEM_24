# %%

import numpy as np
import matplotlib.pyplot as plt
import time

from flax import linen as nn
from flax.training import train_state

from jax import random as jrnd
from jax import numpy as jnp
import jax

from tqdm import trange

import optax

from utils import get_A, get_b, get_coeffs

N = 64
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
k_list = []

# %%
M = 10_000

for i in range(M):
    coeffs = get_coeffs()
    kx = k(x)
    A = get_A(N, kx)
    b = get_b(N)
    u = np.linalg.solve(A, b)
    u_list.append(u)
    coeffs_list.append(coeffs)
    k_list.append(kx)

u_array = jnp.array(u_list)
coeffs_array = (jnp.log10(jnp.array(coeffs_list)) + 2) / 4
k_array = jnp.array(k_list)

# %%
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%

net_width = 128
batchsize = 128
num_iterations = 5_000

class MLP(nn.Module):
  num_hid : int
  num_out : int

  def setup(self):
    self.linear1 = nn.Dense(features=self.num_hid)
    self.linear2 = nn.Dense(features=self.num_hid)
    self.linear3 = nn.Dense(features=self.num_hid)
    self.linear4 = nn.Dense(features=self.num_out)

  def __call__(self, x):
    h = x
    h = self.linear1(h)
    h = nn.swish(h)
    h = self.linear2(h)
    h = nn.swish(h)
    h = self.linear3(h)
    h = nn.swish(h)
    h = self.linear4(h)
    return h

model = MLP(num_hid=net_width, num_out=1) # NN representing s

# %%
print(model)

# %%

initial_learning_rate = 1e-2
final_learning_rate = 1e-6

learning_rate_schedule = optax.cosine_decay_schedule(
    init_value=initial_learning_rate,
    decay_steps=num_iterations,
    alpha=final_learning_rate
)

key = jrnd.PRNGKey(0)
key, p_key = jrnd.split(key)
params = model.init(p_key, jnp.zeros(1+4))
optimizer = optax.adam(learning_rate=learning_rate_schedule)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=optimizer)

# %%

def compute_error_at_index(params, i):
    u_true = u_array[i] # N
    coeffs = coeffs_array[i] # 4
    _x = jnp.linspace(0, 1, N) # N
    _xc = jnp.column_stack((_x, jnp.outer(jnp.ones(N), coeffs))) # N x 5
    u_pred = jax.vmap(lambda x: model.apply(params, x))(_xc) # N x 1
    u_pred = u_pred.reshape(-1) # N
    return jnp.sum((u_true - u_pred)**2)

def loss_fn(state, params, key):
    indices = jrnd.randint(key, (batchsize,), 0, M)
    losses = jax.vmap(lambda i: compute_error_at_index(params, i))(indices)
    return losses.mean()
    
# %%

@jax.jit
def train_step(state, key):
   grad_fn = jax.value_and_grad(loss_fn, argnums=1)
   loss, grads = grad_fn(state, state.params, key)
   state = state.apply_gradients(grads=grads)
   return state, loss
# %%

state, loss = train_step(state, key)

loss_plot = [ ]
key, loop_key = jrnd.split(key)
for iter in trange(num_iterations):
   loop_key, _ = jrnd.split(loop_key)
   state, loss = train_step(state, loop_key)
   loss_plot.append(loss)
# %%

plt.plot(loss_plot)
plt.yscale('log')

# %%
for k in range(10):
    i = np.random.randint(0, M)
    u_true = u_array[i] # N
    coeffs = (coeffs_array[i]) # 4
    _x = jnp.linspace(0, 1, N) # N
    _xc = jnp.column_stack((x, jnp.outer(jnp.ones(N), coeffs))) # Nx5
    u_pred = jax.vmap(lambda x: model.apply(state.params, x))(_xc)

    if k == 0:
        plt.plot(x, u_true, color = 'k', label='True')
        plt.plot(x, u_pred, color = 'c', label='NN approx.')
    else:
        plt.plot(x, u_true, color = 'k')
        plt.plot(x, u_pred, color = 'c')
        
plt.legend()
plt.show()

# %%
