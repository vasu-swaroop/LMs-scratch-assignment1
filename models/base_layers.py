from flax import linen as nn 
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array

from .schemas import Activation

class FFN(nn.Module):
    weights: list[int]
    activation: Activation

    @nn.compact
    def __call__(self, x:Float[Array,'B ... D_in'], last_layer_act=False)->Float[Array, 'B ... D_out']:
        for weight in self.weights[:-1]:
            x = nn.Dense(weight, use_bias=True)(x)
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


class pos_to_freq(nn.Module):
    model_dim:int

    @nn.compact
    def __call__(self, pos:Float[Array, 'B S']):
        freq=-2*pos/self.model_dim
        freq= 10000**freq
        return freq  

class Rope(nn.Module):
    model_dim:int

    @nn.compact
    def __call__(self, x:Float[Array, 'B S H D'], pos:Float[Array, 'S'])-> Float[Array, 'B S H D']:
        pos=pos_to_freq(self.model_dim)(pos[None,:])
        cos_pos=jnp.cos(pos)[:,:,None,None] # B S 
        sin_pos=jnp.sin(pos)[:,:,None,None]
        half_dim=x.shape[-1]//2

        even=x[:,:,:,:half_dim] # 'B S H D//2'
        odd= x[:,:,:,half_dim:]
        assert even.shape[-1]==odd.shape[-1]

        even_=even * cos_pos - odd * sin_pos # 'B S H D//2'
        odd_=even * cos_pos + odd * sin_pos

        return jnp.concat([even_, odd_], axis=-1)

class Attention(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, q:Float[Array, 'B S H D'], k:Float[Array, 'B S H D'], v:Float[Array, 'B S H D'])-> Float[Array, 'B S H D']:
        similarity= similarity_dot_prod(q, k)
        similarity= similarity * self.hidden_dim**(-0.5)
        softmax_scores= nn.softmax(similarity,axis=-1) # B S H D D
        attention = softmax_scores @ v # B S H D
        return attention
    #TODO: Add masking for causal inference as well



def test_transformer_forward():
    print("Testing Transformer")

    rng=jax.random.PRNGKey(42)
    inp =jnp.ones((7,6,4096))
    transformer_block=TransformerBLock(latent_dim=256, hidden_dim=512, num_heads=8, model_dim=4096, activation=Activation.RELU)
    model_params=transformer_block.init(rng, inp)
    out= transformer_block.apply(model_params, inp)
    print(out)
    print(out.shape)

    #TODO: Add an assert, and mathematically compare what will be the value of MLA Transformer when all input is 1

def test_attetnion_forward():
    print("Tessting attention")

    attn=Attention(hidden_dim=512)
    rng=jax.random.PRNGKey(42)
    inp =jnp.ones((1000,100,10,512))
    model_params=attn.init(rng, inp, inp, inp)
    out= attn.apply(model_params, inp,inp,inp)
    print(out)
    print(out.shape)

def test_mlp_forward():
    print("Testing FFN Layers")
    mlp=FFN(weights=[128,64,32,16], activation=Activation.RELU)
    rng=jax.random.PRNGKey(42)
    inp=jnp.ones((1000,128))
    model_var=mlp.init(rng,inp)
    inp=jnp.ones((1000,128))
    out=mlp.apply(model_var, inp)     
    print(out.shape)

def test_rope_forward():
    print("testing rope")
    rope=Rope(12312312)
    inp_seq=jnp.ones((15,14,12,34))
    inp_pos=jnp.arange(0,14)

    var=rope.init(jax.random.PRNGKey(42), inp_seq, inp_pos)
    out=rope.apply(var, inp_seq, inp_pos)

    print(out.shape)

if __name__== "__main__":
    test_rope_forward()