from enum import Enum
from flax import linen as nn
from dataclasses import dataclass
from models.activations import LearnedGelu, SwiGlu, Relu, Gelu, Silu

class Activation(Enum):
    RELU = Relu
    GELU = Gelu
    SWISH = Silu
    LGELU = LearnedGelu
    SWIGLU = SwiGlu
    
class RouterType(Enum):
    LEARNED = "learned"

@dataclass
class MLA_config:
    latent_dim_q:int
    latent_dim_kv:int
    dim_content: int
    dim_pos: int
    num_heads: int

@dataclass
class MOE_FFN_config:
    num_shared_experts: int  # Number of shared experts (always used)
    num_routing_experts: int  # Number of routing experts (to select from)
    num_selected_experts: int  # Top-k experts to select from routing experts
    activation: Activation
    router_type: RouterType

@dataclass
class ModelConfig:
    mla_config: MLA_config
    moe_ffn_config: MOE_FFN_config
    model_dim: int
    transformer_depth: int
    vocab_length: int
    hidden_dim: int | None = None
    num_heads: int | None = None
    activation: Activation | None = None

