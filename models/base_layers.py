from flax import linen as nn 
from jax import numpy as jnp
import jax
from jaxtyping import Float, Array, Int, PRNGKeyArray
from .schemas import Activation, ActivationType

from typing import Optional
def trunc_init(init_key:PRNGKeyArray, weight_shape, dtype=jnp.float32):
    fan_in, fan_out=weight_shape
    #Controls variance
    std=(2/(fan_in+fan_out))**0.5
    weights=jax.random.truncated_normal(init_key, -3*std,3*std, weight_shape, dtype=dtype) 
    return weights

class customDense(nn.Module):
    out_shape: int
    in_shape: Optional[int]=None
    dtype: jnp.dtype= jnp.float32
    @nn.compact
    def __call__(self,x:Float[Array, 'B ... D_in'])->Float[Array, 'B ... D_out']:
        if not self.in_shape:
            in_shape=x.shape[-1]
        weights=self.param('weight', trunc_init,(in_shape, self.out_shape))
        out= jnp.matmul(x, weights)
        return out

class customEmbedding(nn.Module):
    vocab_size:int
    embedding_size:int

    def take_element(self, embed_table, idx):
        return embed_table.at[idx].get()
    @nn.compact
    def __call__(self, x:Int[Array, 'B S'])-> Float[Array, 'B S D']:
        embed_table=self.param('embed',trunc_init, (self.vocab_size, self.embedding_size))
        embeddings= jax.vmap(self.take_element,in_axes=(None,0))(embed_table, x)
        return embeddings
class FFN(nn.Module):
    weights: list[int]
    activation: Activation

    @nn.compact
    def __call__(self, x:Float[Array,'B ... D_in'], last_layer_act=False, weight_init_test=True)->Float[Array, 'B ... D_out']:
        for i, weight in enumerate(self.weights[:-1]):
            x = customDense(weight)(x)
            if self.activation[1]==ActivationType.MODULE:
                x=self.activation[0]()(x)
            else:
                x=self.activation[0](x)
        x =customDense(self.weights[-1])(x)
        if last_layer_act:
            if self.activation[1]==ActivationType.MODULE:
                x=self.activation[0]()(x)
            else:
                x=self.activation[0](x)
        # x=RMSNorm()(x) #TODO original FFN did not have this
        return x

    def __post_init__(self):
        super().__post_init__()
        assert len(self.weights) >= 2, "Need input + output layer"

class RMSNorm(nn.Module):
    d_model: int 
    epsilon: float= 1e-6
    def sqr_sum(self, x:Float[Array,'B ... D'])->Float[Array,'B ... D']:
        return jnp.mean(x**2, axis=-1,keepdims=True)

    @nn.compact
    def __call__(self, x:Float[Array,'B ... D'])->Float[Array,'B ... D']:
        gain=self.param('gain',nn.initializers.constant(1), (self.d_model))
        sq_sum= self.sqr_sum(x)
        denom=jnp.mean(jnp.sqrt(sq_sum),axis=-1)+self.epsilon
        return x /denom[...,None] *gain

def similarity_dot_prod(x:Float[Array, 'B ... D'], y:Float[Array, 'B ... D'])-> Float[Array, 'B ... D D']:
   y=jnp.permute_dims(y, (0,1,3,2))

   return jnp.matmul(x, y)


class pos_to_freq(nn.Module):
    model_dim:int
    theta: int= 10000
    @nn.compact
    def __call__(self, pos:Int[Array, 'B S']):
        freq=-2*pos/self.model_dim
        freq= self.theta**freq
        return freq  

class Rope(nn.Module):
    model_dim:int
    @nn.compact
    def __call__(self, x:Float[Array, '... D'], pos:Float[Array, 'S'])-> Float[Array, '... D']:        
        pos=pos_to_freq(self.model_dim)(pos)
        pos=pos[:,:,None]

        cos_pos=jnp.cos(pos)
        sin_pos=jnp.sin(pos)

        half_dim=x.shape[-1]//2
        even=x[...,:half_dim] # 'B S H D//2' or  B S D//2
        odd= x[...,:,half_dim:]

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

def gumbel_softmax(x:Float[Array, '... D'], key:PRNGKeyArray, temprature:Float=0.1)->Float[Array, '... D']:
    key, subkey=jax.random.split(key)
    gumbel_noise=jax.random.gumbel(subkey, x.shape)
    x=(x+gumbel_noise)/temprature
    x=nn.softmax(x, axis=-1)
    return x, key
    
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
    jax.tree_util.tree_map(lambda x: jnp.zeros_like, model_var['params'])
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

def test_cutom_dense():
    rng=jax.random.PRNGKey(3435)
    inp=jnp.ones((1000,123,128))
    dtype=jnp.bfloat16
    model=customDense(20)
    vars_=model.init(rng, inp)
    out=model.apply(vars_, inp)

def test_custom_embedding():
    rng=jax.random.PRNGKey(3435)
    inp=jnp.ones((1000,128), dtype=jnp.int16)
    model=customEmbedding(2048, 14043)
    vars_= model.init(jax.random.PRNGKey(42), inp)
    out= model.apply(vars_, inp)
    print(out.shape)
# class test_custom_embed():
if __name__== "__main__":
    test_custom_embedding()

