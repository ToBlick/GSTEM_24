# %%
import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random
from typing import Sequence
from utils import generate_data
seed = 0
learning_rate = 0.001
batch_size = 512
x_test, y_test = generate_data(N=1_000, unitless=True, seed=123)
x_test, y_test = x_test.reshape(-1, 1), y_test.reshape(-1, 1)
class MLP(nn.Module):
   neurons_per_layer: Sequence[int]
   @nn.compact
   def __call__(self, x):
      for n in self.neurons_per_layer:
         x = nn.Dense(n)(x)
         x = nn.tanh(x)
      return x
x_key, init_key = random.split(random.PRNGKey(seed))
model = MLP(neurons_per_layer=[10, 10, 10])
x = random.uniform(x_key, (1, 1))
params = model.init(init_key, x)
y = model.apply(params, x)
def model_forward(params, x):
   return model.apply(params, x)
def loss_fn(params, batch):
   model_output = model_forward(params, batch)
   x_test, y_test = batch
   loss = jnp.mean((y_test - model_output)**2)
   return loss
#batch = (x_test, y_test)
#loss = loss_fn(params, batch)
#print(f'Loss: {loss}')
optimiser = optax.adam(learning_rate)
opt_state = optimiser.init(params)
@jax.jit
def train_step(params, batch):
   loss, grads = jax.value_and_grad(loss_fn)(params, batch)
   updates, opt_state = optimiser.update(grads, opt_state, params)
   params = optax.apply_updates(params, updates)
   return params, loss