from flax import linen as nn
from jaxtyping import Float, Array
from jax import numpy as jnp

class LearnedGelu(nn.Module):
    '''Want to experiment with implicit statistical learning of the Gausian the GELU approximates
    This would be like trying to shift the zero/clip point and the strength of non linearity.
    As sigma goes to zero, becomes more and more like Geli'''
    dtype : jnp.dtype = jnp.bfloat16
    @nn.compact
    def __call__(self, inp:Float[Array, '... D'])->Float[Array, '... D']:
        mean=self.param('mean', nn.initializers.constant(0),(),dtype=self.dtype)
        scale=self.param('scale', nn.initializers.constant(1),(),dtype=self.dtype)
        inp=(inp-mean)*scale
        return nn.gelu(inp).astype(self.dtype)

def softmax(x, axis=-1):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

def sigmoid(x: Float[Array, '... D']) -> Float[Array, '... D']:
    return 1 / (1 + jnp.exp(-x))

class Silu(nn.Module):
    def __call__(self, x: Float[Array, '... D']) -> Float[Array, '... D']:
        return x * sigmoid(x)

class Relu(nn.Module):
    def __call__(self, x: Array) -> Array:
        return nn.relu(x)

class Gelu(nn.Module):
    def __call__(self, x: Array) -> Array:
        return nn.gelu(x)

class SwiGlu(nn.Module):
    model_dim: int
    compact_dim: int 
    dtype: jnp.dtype= jnp.bfloat16
    
    @nn.compact
    def __call__(self, x:Float[Array, '... D'])->Float[Array, '... D']:
        outer_up = nn.Dense(self.model_dim, dtype=self.dtype, name="outer_up")
        inner_down_l = nn.Dense(self.compact_dim, dtype=self.dtype, name="inner_down_l")
        inner_down_r = nn.Dense(self.compact_dim, dtype=self.dtype, name="inner_down_r")
        return outer_up(Silu(name="silu")(inner_down_l(x)) * inner_down_r(x))