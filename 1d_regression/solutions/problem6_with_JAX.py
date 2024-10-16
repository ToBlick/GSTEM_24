#%%
import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random
from typing import Sequence
from utils import generate_data_jax

from tqdm import trange

import matplotlib.pyplot as plt

seed = 0  
learning_rate = 0.001
batch_size = 512
num_iterations = 5_000

# %%
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%

def getBatch(key):
   x_train, y_train = generate_data_jax(key, N=batch_size)
   x_train, y_train = x_train.reshape(-1, 1), y_train.reshape(-1, 1)
   batch = jnp.column_stack((x_train, y_train))
   return batch

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
    h = nn.tanh(h)
    h = self.linear2(h)
    h = nn.tanh(h)
    h = self.linear3(h)
    h = nn.tanh(h)
    h = self.linear4(h)
    return h
#%%
x_key, init_key = random.split(random.PRNGKey(seed))

model = MLP(num_hid=10, num_out=1)
print(model)
#%%
#x = random.uniform(x_key, (1, 1))
#y = model.apply(params, x)

def model_forward(params, x):
   return model.apply(params, x)

def loss_fn(state, params, key):
   batch = getBatch(key)
   x_test = batch[:,0]
   y_test = batch[:,1]
   x_test = x_test.reshape(-1, 1)
   y_test = y_test.reshape(-1, 1)
   model_output = jax.vmap(lambda x: model_forward(params, x))(x_test)
   loss = jnp.sum((y_test - model_output)**2)
   return loss

#batch = (x_test, y_test)
#loss = loss_fn(params, batch)
#print(f'Loss: {loss}')

params = model.init(init_key, jnp.zeros(1))
optimizer = optax.adam(learning_rate=learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=optimizer)

@jax.jit
def train_step(state, key):
   grad_fn = jax.value_and_grad(loss_fn, argnums=1)
   loss, grads = grad_fn(state, state.params, key)
   state = state.apply_gradients(grads=grads)
   return state, loss

# %%

key, loc_key = random.split(x_key)
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

# %%

x_test, y_test = generate_data_jax(random.PRNGKey(123), N=1_000)
x_test = x_test.reshape(-1, 1)

plt.scatter(x_test, y_test, label='True data', color='k', s=1)
plt.scatter(x_test, jax.vmap(lambda x: model_forward(state.params, x))(x_test), label='Model predictions', color = 'c', s=1)
plt.legend()
plt.show()
# %%
