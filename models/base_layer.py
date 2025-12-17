from flax import linen as nn 
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array

from schemas import Activation

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

class Attention(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, q:Float[Array, 'B S H D'], k:Float[Array, 'B S H D'], v:Float[Array, 'B S H D'])-> Float[Array, 'B S H D']:
        similarity= similarity_dot_prod(q, k)
        similarity= similarity * self.hidden_dim**(-0.5)
        softmax_scores= nn.softmax(similarity,axis=-1) # B S H D D
        attention = softmax_scores @ v # B S H D
        return attention


class MLA(nn.Module):
    '''Currently implementing non rope, non kv cache implementation'''
    latent_dim:int
    hidden_dim:int
    num_heads: int
    model_dim: int
    @nn.compact
    def to_kv_tokens(self, x:Float[Array, 'B S M'])-> Float[Array, 'B S H D']:
        cache = nn.Dense(self.latent_dim)(x) # 'B S d'
        x = nn.Dense(self.num_heads*self.hidden_dim)(cache) # 'B S H*D'
        x = jnp.reshape(x, (x.shape[0],x.shape[1], self.num_heads, self.hidden_dim))
        return x, cache
    
    @nn.compact
    def to_q_tokens(self,x:Float[Array, 'B S M'])-> Float[Array, 'B S H D']:
        x = nn.Dense(self.hidden_dim*self.num_heads)(x)
        x = jnp.reshape(x, (x.shape[0],x.shape[1], self.num_heads, self.hidden_dim))
        return x

    @nn.compact
    def __call__(self, x:Float[Array, 'B S M'])-> Float[Array, 'B S M']:
        kv_concat=jnp.stack([x,x], axis=0)
        kv_tokens, kv_cache=jax.vmap(self.to_kv_tokens)(kv_concat) #TODO: implement KV cache
        k_tokens, v_tokens= jnp.split(kv_tokens,2, axis=0)
        k_tokens, v_tokens= k_tokens[0], v_tokens[0]

        q_tokens= self.to_q_tokens(x)
        
        attention=Attention(self.hidden_dim)(q_tokens, k_tokens, v_tokens)
        
        #Concatenate the multiple heads
        attention= jnp.reshape(attention, (attention.shape[0],attention.shape[1], self.num_heads* self.hidden_dim))

        #Up project
        out_token=nn.Dense(self.model_dim)(attention)
        return out_token


''' Inpsired from https://arxiv.org/pdf/2412.19437'''
class TransformerBLock(nn.Module):
    latent_dim:int
    hidden_dim:int
    num_heads: int
    model_dim:int
    activation: Activation

    @nn.compact
    def __call__(self, x:Float[Array, 'B S D'])-> Float[Array,'B S D']:
        stream=x
        x=RMSNorm()(x)
        x=MLA(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, model_dim=self.model_dim)(x)
        stream=x+stream
        x=RMSNorm()(stream)
        x=FFN(weights=[self.model_dim, self.model_dim*4, self.model_dim], activation=self.activation)(x,last_layer_act=True)
        stream=x+stream
        return stream


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
    print("Tessting atte4ntion")
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

if __name__== "__main__":
    test_transformer_forward()