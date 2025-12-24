from .base_layer import RMSNorm, MLA, FFN
from flax import linen as nn
from jaxtyping import Float, Array, PRNGKeyArray, Int
from typing import Union
from tokenizer.tokens import Token
from tokenizer.tokenizer import Tokenizer
from .schemas import ModelConfig, Activation
from jax import numpy as jnp
import jax

from pathlib import Path

''' Inpsired from https://arxiv.org/pdf/2412.19437'''
class TransformerBlock(nn.Module):
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

class Embedding(nn.Module):
    vocab_length:int
    model_dim: int

    @nn.compact
    def __call__(self, x:Float[Array, 'B S'])-> Float[Array, 'B S D']:
        embed=nn.Embed(self.vocab_length,self.model_dim)
        x=embed(x)
        return x

class DeepSeekModel(nn.Module):
    model_config: ModelConfig

    @nn.compact
    def __call__(self, token_idx_list: Int[Array, 'B S'])->Float[Array, 'B S V']:

        emebeddings=Embedding(self.model_config.vocab_length, self.model_config.model_dim)(token_idx_list)
        x=jnp.stack(emebeddings)
        BlockStack = nn.scan(
            TransformerBlock,
            variable_axes={"params": 0},   # ← separate params per depth
            split_rngs={"params": True},
            variable_broadcast=False,
            length=self.model_config.transformer_depth,
        )

        x,_ = BlockStack(
            latent_dim=self.model_config.latent_dim,
            hidden_dim=self.model_config.hidden_dim,
            num_heads=self.model_config.num_heads,
            model_dim=self.model_config.model_dim,
            activation=self.model_config.activation,
        )(x)
        x= nn.Dense(self.model_config.vocab_length)(x)
        x= nn.softmax(x,axis=-1)
        print(x.shape)
        return x


def test_transformer_forward():
    model_config=ModelConfig(   
        latent_dim=8,
        hidden_dim=128,
        num_heads=4, 
        model_dim=256,
        activation= Activation.RELU,
        transformer_depth=4,
        vocab_length=32_000)
    checkpoint_path= Path('/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl')
    
    
    tokenizer = Tokenizer.load_for_inference(checkpoint_path)
    model=DeepSeekModel(model_config=model_config,)
    input_data= jnp.asarray(range(10000))
    print(input_data.shape)
    input_data= jnp.expand_dims(input_data, axis=0)
    key=jax.random.PRNGKey(42)
    variables=model.init(key, input_data)
    model.apply(variables, input_data)  

if __name__=='__main__':

    test_transformer_forward()