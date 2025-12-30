from models.base_layers import RMSNorm
from models.attention import MLA_rope, MOE_FFN 
from flax import linen as nn
from jaxtyping import Float, Array, PRNGKeyArray, Int
from typing import Union
from tokenizer.tokens import Token
from tokenizer.tokenizer import Tokenizer
from models.schemas import ModelConfig, Activation, MLA_config, MOE_FFN_config, RouterType
from jax import numpy as jnp
import jax
from models.base_layers import customDense, customEmbedding

from pathlib import Path

''' Inpsired from https://arxiv.org/pdf/2412.19437'''
class TransformerBlock(nn.Module):
    mla_config: MLA_config
    model_dim:int
    moe_ffn_config: MOE_FFN_config

    @nn.compact
    def __call__(self, x:Float[Array, 'B S D'])-> Float[Array,'B S D']:
        stream=x
        x=RMSNorm(self.model_dim)(x)
        pos=jnp.arange(x.shape[1])[None, :].repeat(x.shape[0], 0)
        x=MLA_rope(self.mla_config, self.model_dim)(x,use_cache=False, pos=pos)
        stream=x+stream
        x=RMSNorm(self.model_dim)(stream)
        x=MOE_FFN(config=self.moe_ffn_config, model_dim=self.model_dim)(x)
        stream=x+stream
        return stream, 1




class Sampling():
    pass

class DeepSeekModel(nn.Module):
    model_config: ModelConfig

    # @jax.jit
    @nn.compact
    def __call__(self, token_idx_list: Int[Array, 'B S'])->Float[Array, 'B S V']:

        x=customEmbedding(self.model_config.vocab_length, self.model_config.model_dim)(token_idx_list)
        BlockStack = nn.scan(
            TransformerBlock,
            variable_axes={"params": 0},   
            split_rngs={"params": True, "gumbel": True},
            variable_broadcast=False,
            length=self.model_config.transformer_depth,
        )

        x,_ = BlockStack(
            mla_config=self.model_config.mla_config,
            model_dim=self.model_config.model_dim,
            moe_ffn_config=self.model_config.moe_ffn_config,
        )(x)
        x= RMSNorm(self.model_config.model_dim)(x)
        x= customDense(self.model_config.vocab_length)(x)
        return x


def test_transformer_forward():

    mla_config= MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )

    moe_ffn_config=MOE_FFN_config(
        num_shared_experts=2,
        num_routing_experts=6,
        num_selected_experts=2,
        activation=Activation.RELU,
        router_type=RouterType.LEARNED
    )

    model_config=ModelConfig(   
        mla_config=mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=512,
        transformer_depth=10,
        hidden_dim=128,
        num_heads=8, 
        activation= Activation.RELU,
        vocab_length=32_000)
    checkpoint_path= Path('/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl')
    
    
    tokenizer = Tokenizer.load_for_inference(checkpoint_path)
    model=DeepSeekModel(model_config=model_config,)
    input_data= jnp.asarray(range(10000))
    input_data= jnp.expand_dims(input_data, axis=0)
    key=jax.random.PRNGKey(42)
    variables=model.init({'params': key, 'gumbel': key}, input_data)
    model.apply(variables, input_data, rngs={'gumbel': key})

if __name__=='__main__':

    test_transformer_forward()