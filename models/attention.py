from models.base_layers import Attention
from jax import numpy as jnp
from jaxtyping import Float,Array, Int
import jax 
from models.schemas import MLA_config
from flax import linen as nn
from models.base_layers import Rope
from einops import rearrange
class MLA_rope(nn.Module):
    '''Currently implementing non rope, non kv cache implementation'''
    config: MLA_config
    model_dim: int

    def setup(self):
        config=self.config
        self.Q_d=nn.Dense(config.latent_dim_q) # B S D-> B S d
        self.KV_d=nn.Dense(config.latent_dim_kv) #  B S D-> B S d'

        self.Q_u_c=nn.Dense(config.dim_content*config.num_heads) # B S d-> B S (h*c)
        self.K_u_c=nn.Dense(config.dim_content) # B S d-> B S (h*c)

        self.Q_u_p=nn.Dense(config.dim_pos*config.num_heads) # B S d-> B S (h*p)
        self.K_u_p=nn.Dense(config.dim_pos) # B S d-> B S (h*p)
        # NOTE: In the paper, the Rope for K is only done once, (possibly for benefit of caching)

        self.V_U=nn.Dense(config.dim_content) # B S d -> B S (h*c)
        self.rope=Rope(self.model_dim)

        hidden_dim=self.config.dim_content+self.config.dim_pos

        self.up_proj= nn.Dense(self.model_dim)

        self.attention=Attention(hidden_dim)
        
    def split_to_head(self, x:Float[Array, 'B S D'])-> Float[Array, 'B S h D']:
        x=jnp.reshape(x, (x.shape[0],x.shape[1], self.config.num_heads, -1))
        return x

    def kv_to_heads(self,x:Float[Array, 'B S D'])->Float[Array, 'B S hD']:
        x=jnp.repeat(x, self.config.num_heads, axis=-1)
        return x

    def __call__(self, x:Float[Array, 'B S D'], use_cache:bool=False, pos:Int[Array, 'B S']=None)-> Float[Array, 'B S D']:
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

        if use_cache:
            kv_latent_cached= self.cache["kv_latent"] # B S_old d'
            k_r_Cached= self.cache["k_rope"] #  B S_old d_p,

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
        attention=rearrange(attention, 'B S h d-> B S (h d)')
        #Up project
        out_token=self.up_proj(attention)
        return out_token+x



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
    test_mla_forward()