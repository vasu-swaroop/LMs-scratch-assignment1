from models.base_layers import Attention, gumbel_softmax,Rope, FFN
from jax import numpy as jnp
from jaxtyping import Float, Array, Int
import jax 
from models.schemas import MLA_config, MOE_FFN_config, RouterType
from flax import linen as nn
from models.schemas import Activation
from einops import rearrange
from models.base_layers import customDense
class MLA_rope(nn.Module):
    '''Currently implementing non rope, non kv cache implementation'''
    config: MLA_config
    model_dim: int


    def setup(self):
        config=self.config
        self.Q_d=customDense(config.latent_dim_q) # B S D-> B S d
        self.KV_d=customDense(config.latent_dim_kv) #  B S D-> B S d'

        self.Q_u_c=customDense(config.dim_content*config.num_heads) # B S d-> B S (h*c)
        self.K_u_c=customDense(config.dim_content) # B S d-> B S (h*c)

        self.Q_u_p=customDense(config.dim_pos*config.num_heads) # B S d-> B S (h*p)
        self.K_u_p=customDense(config.dim_pos) # B S d-> B S (h*p)
        # NOTE: In the paper, the Rope for K is only done once, (possibly for benefit of caching)

        self.V_U=customDense(config.dim_content) # B S d -> B S (h*c)
        self.rope=Rope(self.model_dim)

        hidden_dim=self.config.dim_content+self.config.dim_pos

        self.up_proj= customDense(self.model_dim)

        self.attention=Attention(hidden_dim)
        
    def split_to_head(self, x:Float[Array, 'B S D'])-> Float[Array, 'B S h D']:
        x=jnp.reshape(x, (x.shape[0],x.shape[1], self.config.num_heads, -1))
        x =rearrange(x, 'B S h D-> B h S D')
        return x

    def kv_to_heads(self,x:Float[Array, 'B S D'])->Float[Array, 'B S hD']:
        x=jnp.repeat(x, self.config.num_heads, axis=-1)
        return x
    
    def __call__(self, x:Float[Array, 'B S D'], build_cache:bool=False, use_cache:bool=False, pos:Int[Array, 'B S']=None,seq_idx:int=0)-> Float[Array, 'B S D']:
        B,S,D= x.shape
        if use_cache:
            assert S==1, "Cache only supported during inference"
        #NOTE: in case of inference, the Sequence should be 1

        q_latent= self.Q_d(x) # B S l
        q_c =self.Q_u_c(q_latent) # B S (h d_c)
        q_r =self.Q_u_p(q_latent) # B S (h d_p)
        q_r=self.rope(q_r, pos) # B S (h d_p)
        q= jnp.concat([q_c, q_r], axis=-1) # B S( h (d_c + d_p))


        kv_latent=self.KV_d(x) # B S d' , B 1 d' for inf
        k_r=self.K_u_p(kv_latent) # B S d_p,  B 1 d_p for inf
        k_r=self.rope(k_r, pos) # B S (d_p)

        if build_cache:
            NUM_BLOCKS=8
            BLOCK_SIZE=128
            kv_latent_cache=jnp.zeros(NUM_BLOCKS, BLOCK_SIZE,self.config.latent_dim_kv)
            k_r_cache=jnp.zeros(NUM_BLOCKS, BLOCK_SIZE,self.config.latent_dim_kv)
            
            kv_latent_cached= self.cache["kv_latent"] # B S_old d'
            k_r_Cached= self.cache["k_rope"] #  B S_old d_p,
            

        if use_cache:
            #Change latents
            kv_latent= jnp.concat([kv_latent_cached, kv_latent], axis=1) # B S_old+1, d'
            k_r= jnp.concat([k_r_Cached, k_r], axis=1) # B S_old+1, d_p

        k_c=self.K_u_c(kv_latent) # B S (d_c)
        k=jnp.concat([k_c, k_r], axis=-1) # B S (d_c + d_p)
        v=self.V_U(kv_latent)
        k=self.kv_to_heads(k)
        v=self.kv_to_heads(v)

        q_tokens, k_tokens, v_tokens=self.split_to_head(q), self.split_to_head(k), self.split_to_head(v) # B S h d
        #TODO: Fuse weights in attention

        
        attention=self.attention(q_tokens, k_tokens, v_tokens)
        attention=rearrange(attention, 'B h S  d-> B S (h d)')
        #Up project
        out_token=self.up_proj(attention)
        return out_token+x

class MOE_FFN(nn.Module):
    config: MOE_FFN_config
    model_dim:int
    dtype: jnp.dtype | None = jnp.bfloat16

    def setup(self):
        self.shared_experts = nn.vmap(
            FFN,
            variable_axes={"params": 0}, 
            split_rngs={"params": True}, 
            in_axes=None,
            out_axes=0,
            axis_size=self.config.num_shared_experts,
            axis_name="experts"
        )(
            weights=[self.model_dim, self.model_dim*4, self.model_dim],
            activation=self.config.activation
        )
        
        self.routing_experts = nn.vmap(
            FFN, 
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.config.num_routing_experts
        )(
            weights=[self.model_dim, self.model_dim*4, self.model_dim],
            activation=self.config.activation
        )

        self.router = FFN([self.model_dim, self.config.num_routing_experts], activation=self.config.activation)

    def __call__(self, x:Float[Array, 'B S D'])->Float[Array, 'B S D']:
        shared_out = self.shared_experts(x)
        all_experts = self.routing_experts(x)

        logits=self.router(x)
        key = self.make_rng('gumbel')
        router_probs, _=gumbel_softmax(logits, key, temprature=0.1) 
        router_probs=router_probs.astype(self.dtype)
        chosen_experts=jax.lax.top_k(router_probs, k=self.config.num_selected_experts, axis=-1)[1]
        
        one_hot_vectors = jax.nn.one_hot(chosen_experts, self.config.num_routing_experts)
        one_hot_vectors=one_hot_vectors.astype(self.dtype)
        out=jnp.einsum("B S c O,O B S D->c B S D", one_hot_vectors, all_experts)
        out=jnp.sum(out, axis=0)
        
        shared_out_sum = jnp.sum(shared_out, axis=0)
        return out + shared_out_sum 

def test_MOE():
    print("Testing MOE")
    rng=jax.random.PRNGKey(42)
    inp =jnp.ones((7,6,4096))
    out,key=jax.random.split(rng)

    moe_ffn_config=MOE_FFN_config(
    num_shared_experts=2,
    num_routing_experts=6,
    num_selected_experts=2,
    activation=Activation.RELU,
    router_type=RouterType.LEARNED
    )

    model_params=MOE_FFN(moe_ffn_config, model_dim=4096).init({'params': key, 'gumbel': key}, inp)
    out=MOE_FFN(moe_ffn_config, model_dim=4096).apply(model_params, inp,rngs={'gumbel': key})
    print(out.shape)

def test_mla_forward():
    print("Testing mla")
    rng=jax.random.PRNGKey(42)
    inp =jnp.ones((7,6,4096))
    pos =jnp.ones((7,6))
    mla_config = MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )
    mla_rope = MLA_rope(config=mla_config, model_dim=4096)
    rng=jax.random.PRNGKey(42)
    model_params=mla_rope.init(rng, inp, False,pos)
    out= mla_rope.apply(model_params, inp,False, pos)
    #TODO: Add an assert, and mathematically compare what will be the value of MLA Transformer when all input is 1
    print(out.shape, out)
    # assert jnp.equal(out, inp).all()
if __name__== '__main__':
    test_MOE()