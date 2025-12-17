from flax import linen as nn 
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array

from schemas import Activation

class Linear(nn.Module):
    weights: list[int]
    activation: Activation

    @nn.compact
    def __call__(self, x:Float[Array,'B ... D_in'], last_layer_act=False)->Float[Array, 'B ... D_out']:
        for weight in self.weights[:-1]:
            x = nn.Dense(weight, use_bias=True,)(x)
            x = self.activation(x)
        x =nn.Dense(self.weights[-1])(x)
        if last_layer_act:
            x=self.activation(x)

        x=RMSNorm()(x)
        return x

    def __post_init__(self):
        super().__post_init__()
        assert len(self.weights) >= 2, "Need input + output layer"
        assert all(w > 0 for w in self.weights)

class RMSNorm(nn.Module):
    epsilon: float= 1e-6

    def sqr_sum(self, x):
        return jnp.mean(x**2, axis=-1,keepdims=True)

    @nn.compact
    def __call__(self, x:Float[Array,'B ... D']):
        sq_sum= self.sqr_sum(x)
        return x / (jnp.sqrt(sq_sum) +self.epsilon)

def similarity_dot_prod(x:Float[Array, 'B ... D'], y:Float[Array, 'B ... D'])-> Float[Array, 'B ... D D']:
   y=jnp.permute_dims(y, (0,1,3,2))

   return jnp.matmul(x, y)

class Attention(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, q:Float[Array, 'B S H D'], k:Float[Array, 'B S H D'], v:Float[Array, 'B S H D'])-> Float[Array, 'B S H D']:
        similarity= similarity_dot_prod(q, k)
        similarity= similarity * self.hidden_dim**(-0.5)
        softmax_scores= nn.softmax(similarity,axis=-1) # B S H D D
        attention = softmax_scores @ v # B S H D
        return attention


def test_attetnion_forward():
    print("Tessting atte4ntion")
    attn=Attention(hidden_dim=512)
    rng=jax.random.PRNGKey(42)
    inp =jnp.ones((1000,100,10,512))
    model_params=attn.init(rng, inp, inp, inp)
    out= attn.apply(model_params, inp,inp,inp)
    print(out)
    print(out.shape)

def test_mlp_forward():
    print("Testing Linear Layers")
    mlp=Linear(weights=[128,64,32,16], activation=Activation.RELU)
    rng=jax.random.PRNGKey(42)
    inp=jnp.ones((1000,128))
    model_var=mlp.init(rng,inp)
    inp=jnp.ones((1000,128))
    out=mlp.apply(model_var, inp)     
    print(out.shape)

if __name__== "__main__":
    test_attetnion_forward()