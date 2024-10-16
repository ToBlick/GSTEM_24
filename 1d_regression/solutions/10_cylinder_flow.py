#%%
import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random
from typing import Sequence
from utils import generate_data_jax

from ipywidgets import FloatSlider, interactive, VBox, Output
from IPython.display import display

import numpy as np

from tqdm import trange

import matplotlib.pyplot as plt

# %%
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


# %%
data = np.load('vorticity_array.npy')
times = jnp.linspace(0, 1, len(data))

data = jnp.array(data[100:3200:5, :, 130:])
#times = times[100::10]
Nt, Ny, Nx = data.shape

plt.imshow(data[-1])

# %%

seed = 0  
learning_rate = 0.0001
xy_batch_size = 128
t_batch_size = 64
num_iterations = 25_000
net_size = 64

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

model = MLP(num_hid=net_size, num_out=1)
print(model)
# %%

key = random.PRNGKey(0)
key, init_key = random.split(key)

initial_learning_rate = 1e-2
final_learning_rate = 1e-6

learning_rate_schedule = optax.cosine_decay_schedule(
    init_value=initial_learning_rate,
    decay_steps=num_iterations,
    alpha=final_learning_rate
)

key = random.PRNGKey(0)
key, p_key = random.split(key)
params = model.init(p_key, jnp.zeros(3))
#optimizer = optax.adam(learning_rate=learning_rate_schedule)
optimizer = optax.adam(learning_rate=1e-4)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=optimizer)

# %%

def vorticity_net(params, t, xy):
    x, y = xy
    txy = jnp.column_stack((t, x, y))
    return model.apply(params, txy)

def true_vorticity(t, xy):
    x, y = xy
    t_idx = (t * Nt).astype(int)
    x_idx = (x / 4 * Nx).astype(int)
    y_idx = (y * Ny).astype(int)
    return data[t_idx, y_idx, x_idx]

def pointwise_difference(params, t, xy):
    return (vorticity_net(params, t, xy) - true_vorticity(t, xy))**2

# %%

def loss_fn(state, params, key):
    x_key, y_key, t_key = random.split(key, 3) 
   
    xs = random.uniform(x_key, (xy_batch_size,)) * 4
    ys = random.uniform(y_key, (xy_batch_size,))
    xys = jnp.column_stack((xs, ys))
    ts = random.uniform(t_key, (t_batch_size,))
    
    difference = jax.vmap(lambda t: jax.vmap(lambda xy: pointwise_difference(params, t, xy))(xys))(ts)
    
    return jnp.mean(difference)
# %%

@jax.jit
def train_step(state, key):
   grad_fn = jax.value_and_grad(loss_fn, argnums=1)
   loss, grads = grad_fn(state, state.params, key)
   state = state.apply_gradients(grads=grads)
   return state, loss

# %%
key, loc_key = random.split(key)
state, loss = train_step(state, loc_key)

# %%
loss_plot = [ ]
key, loop_key = random.split(key)
for iter in trange(num_iterations):
   loop_key, _ = random.split(loop_key)
   state, loss = train_step(state, loop_key)
   loss_plot.append(loss)

# %%

plt.plot(loss_plot)
plt.yscale('log')
plt.show()
# %%
t_pred = 0.9

X, Y = np.meshgrid(np.linspace(0, 4, Nx), np.linspace(0, 1, Ny))

XY = jnp.column_stack((X.flatten(), Y.flatten()))

vort_pred = jax.vmap(lambda xy: vorticity_net(state.params, t_pred, xy))(XY)
vort_pred = vort_pred.reshape(Ny, Nx)

plt.imshow(vort_pred)

# %%
t0 = 0
# Create ipywidgets sliders 
t_slider = FloatSlider(value=t0, min=0, max=1, step=0.01, description='t')

# Output widget to display the plot
out = Output()

# Create the initial plot to get the limits
lim = 0.05
fig, ax = plt.subplots(nrows=2)
ax[0].imshow(data[int(t_pred * Nt)], vmin=-lim, vmax=lim)
ax[1].imshow(vort_pred, vmin=-lim, vmax=lim)
plt.close(fig)

def update(t):
    with out:
        out.clear_output(wait=True)
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(data[int(t * Nt)], vmin=-lim, vmax=lim)
        vort_pred = jax.vmap(lambda xy: vorticity_net(state.params, t, xy))(XY)
        vort_pred = vort_pred.reshape(Ny, Nx)
        ax[1].imshow(vort_pred, vmin=-lim, vmax=lim)
        plt.show()

interactive_plot = interactive(update, t=t_slider)
display(VBox([t_slider, out]))

# display
update(t0)

