from .base_layers import Attention
from jax import numpy as jnp
import jax 
from .schemas import MLA_config
from flax import linen as nn

class MLA(nn.Module):
    '''Currently implementing non rope, non kv cache implementation'''
    config: MLA_config
    num_heads: int
    model_dim: int

    def set_weights(self):
        config=self.config

        self.Q_d=nn.Dense(config.latent_dim_q) # B S D-> B S d
        self.KV_d=nn.Dense(config.latent_dim_kv) #  B S D-> B S d'

        self.Q_u_c=nn.Dense(config.dim_content*config.num_heads) # B S d-> B S (h*c)
        self.K_u_c=nn.Dense(config.dim_content*config.num_heads) # B S d-> B S (h*c)

        self.Q_u_p=nn.Dense(config.dim_pos*config.num_heads) # B S d-> B S (h*p)
        self.K_u_p=nn.Dense(config.dim_pos) # B S d-> B S (h*p)
        # NOTE: In the paper, the Rope for K is only done once, (possibly for benefit of caching)

        self.V_U=nn.Dense(config.num_heads*config.dim_content) # B S d -> B S (h*c)

    def split_to_head(self, x:Float[Array, 'B S D'])-> Float[Array, 'B S h D']:
        x=jnp.reshape(x, (x.shape[0],x.shape[1], self.num_heads, self.hidden_dim))
        return x

    @nn.compact
    def __call__(self, x:Float[Array, 'B S D'], use_cache=False)-> Float[Array, 'B S D']:

        B,S,D= x.shape
        if use_cache:
            assert S==1, "Cache only supported during inference"
        #NOTE: in case of inference, the Sequence should be 1

        q_latent= self.Q_d(x) # B S l
        q_c =self.Q_u_c(q_latent) # B S h d_c
        q_r =self.Q_u_p(q_latent) # B S h d_p
        q= jnp.concat([q_c, q_r], axis=-1) # B S h (d_c + d_p)

        
        cache=self.variable(
            'cache',
            'kv_rope',
            lambda: {
                "kv_latent": jnp.zeros((B, 0, self.config.latent_dim_kv)),
                "k_rope": jnp.zeros((B, 0, self.config.dim_pos)),
            },
        )

        kv_latent=self.KV_d(x) # B S d'

        if use_cache:
            kv_latent_cached= cache["kv_latent"]
            k_r_Cached= cache["k_rope"]
            kv_latent= jnp.concat([kv_latent_cached, kv_latent], axis=1)
            k_r_Cached= jnp.concat([k_r_Cached, k_r], axis=1)

        k_c=self.K_u_c(kv_latent) # B S h d_c
        k_r=self.K_u_p(kv_latent) # B S d_p
        k=jnp.concat([k_c, k_r[:,:,None, :].repeat(1,1,self.config.num_heads,1)], axis=-1)  # B S h (dc + d_p)

        k_l_c = nn.Dense(config.dim_content)(kv_latent)
        k_l_p = nn.Dense(config.dim_pos)(kv_latent)

        k_l_p_r= self.apply_rope(k_l_p)
        q_l_p_r= self.apply_rope(q_l_p)

        q= jnp.concat([q_l_c, q_l_p_r],axis=-1)
        k= jnp.concat([k_l_c, k_l_p_r],axis=-1)
        
        q_tokens, k_tokens, v_tokens=self.split_to_head(q), self.split_to_head(k), self.split_to_head(v)

        attention=Attention(self.hidden_dim)(q_tokens, k_tokens, v_tokens)


        # kv_concat=jnp.stack([x,x], axis=0)
        # kv_tokens, kv_cache=jax.vmap(self.to_kv_tokens)(kv_concat) #TODO: implement KV cache
        # k_tokens, v_tokens= jnp.split(kv_tokens,2, axis=0)
        # k_tokens, v_tokens= k_tokens[0], v_tokens[0]

        # q_tokens= self.to_q_tokens(x)
        
        # attention=Attention(self.hidden_dim)(q_tokens, k_tokens, v_tokens)
        
        # #Concatenate the multiple heads
        # attention= jnp.reshape(attention, (attention.shape[0],attention.shape[1], self.num_heads* self.hidden_dim))

        # #Up project
        # out_token=nn.Dense(self.model_dim)(attention)
        return out_token
