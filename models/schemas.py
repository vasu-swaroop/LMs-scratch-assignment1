from enum import Enum
from flax import linen as nn
from dataclasses import dataclass

class Activation(Enum):
    RELU = nn.relu
    GELU = nn.gelu
    SWISH = nn.swish
    # SWIGLU
    def __call__(self, x):
        return self.value(x)

@dataclass
class MLA_config():
    latent_dim_q:int
    latent_dim_kv:int
    dim_content: int
    dim_pos: int
    rope: bool
    num_heads: int

    @property
    def hidden_dim(self):
        return self.dim_content + self.dim_pos

@dataclass
class ModelConfig:
    latent_dim:int
    hidden_dim:int
    num_heads: int
    model_dim:int
    activation: Activation
    transformer_depth: int
    vocab_length: int

