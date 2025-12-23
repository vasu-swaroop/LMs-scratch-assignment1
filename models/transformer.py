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

from pathlib import Path

''' Inpsired from https://arxiv.org/pdf/2412.19437'''
class TransformerBlock(nn.Module):
    mla_config: MLA_config
    model_dim:int
    moe_ffn_config: MOE_FFN_config

    @nn.compact
    def __call__(self, x:Float[Array, 'B S D'])-> Float[Array,'B S D']:
        stream=x
        x=RMSNorm()(x)
        pos=jnp.arange(x.shape[1])[None, :].repeat(x.shape[0], 0)
        x=MLA_rope(self.mla_config, self.model_dim)(x,use_cache=False, pos=pos)
        stream=x+stream
        x=RMSNorm()(stream)
        x=MOE_FFN(config=self.moe_ffn_config, model_dim=self.model_dim)(x)
        stream=x+stream
        return stream, 1


class Embedding(nn.Module):
    vocab_length:int
    model_dim:int
        # self.embeddding_dict:dict[int, Float[Array, 'D']]={}

    # def load_embedding(self, embedding_path:Path):
    #     if embedding_path:
    #         with open(embedding_path, 'rb') as f:
    #             self.embedding_dict=pickle.load(f)

    # def save_embedding(self, embedding_path:Path):
    #     if embedding_path:
    #         with open(embedding_path, 'wb') as f:
    #             pickle.dump(self.embedding_dict, f)
    
    # def init_emebddings(self, key:PRNGKeyArray,  token_list: list[int]):
    #     for token in token_list:
    #         key, subkey=jax.random.split(key)
    #         self.embeddding_dict[token]=jax.random.normal(key, (self.model_dim))

    # def lookup(self, token:int):
    #     return self.__call__(token)

    @nn.compact
    def __call__(self, token_idx_list: Int[Array, 'B S']):
        embed = self.param(
            "embedding",
            nn.initializers.normal(),
            (self.vocab_length, self.model_dim),
        )
        return embed[token_idx_list]
class Sampling():
    pass

class DeepSeekModel(nn.Module):
    model_config: ModelConfig
    # embedding_model: Embedding
    # tokenizer: Tokenizer
    @nn.compact
    def __call__(self, token_idx_list: Int[Array, 'B S'])->Float[Array, 'B S V']:
        print(token_idx_list)
        emebeddings=Embedding(self.model_config.vocab_length, self.model_config.model_dim)(token_idx_list)
        x=jnp.stack(emebeddings)

        BlockStack = nn.scan(
            TransformerBlock,
            variable_axes={"params": 0},   # ← separate params per depth
            split_rngs={"params": True, "gumbel": True},
            variable_broadcast=False,
            length=self.model_config.transformer_depth,
        )

        x,_ = BlockStack(
            mla_config=self.model_config.mla_config,
            model_dim=self.model_config.model_dim,
            moe_ffn_config=self.model_config.moe_ffn_config,
        )(x)
        x= nn.Dense(self.model_config.vocab_length)(x)
        x= nn.softmax(x,axis=-1)
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
        expert_dim=1024,
        activation=Activation.RELU,
        router_type=RouterType.LEARNED
    )

    model_config=ModelConfig(   
        mla_config=mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=1028,
        transformer_depth=2,
        vocab_length=10)

    tokenizer= Tokenizer()
    model=DeepSeekModel(model_config=model_config,)
    input_data= jnp.asarray([1,2,3,4,5])
    input_data= jnp.expand_dims(input_data, axis=0)
    key=jax.random.PRNGKey(42)
    variables=model.init({'params': key, 'gumbel': key}, input_data)
    model.apply(variables, input_data, rngs={'gumbel': key})

if __name__=='__main__':

    test_transformer_forward()