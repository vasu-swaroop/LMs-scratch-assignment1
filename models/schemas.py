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

class RouterType(Enum):
    LEARNED = "learned"

@dataclass
class MLA_config():
    latent_dim_q:int
    latent_dim_kv:int
    dim_content: int
    dim_pos: int
    num_heads: int

@dataclass
class MOE_FFN_config():
    num_shared_experts: int  # Number of shared experts (always used)
    num_routing_experts: int  # Number of routing experts (to select from)
    num_selected_experts: int  # Top-k experts to select from routing experts
    expert_dim: int
    activation: Activation
    router_type: RouterType

@dataclass
class ModelConfig:
    mla_config: MLA_config
    moe_ffn_config: MOE_FFN_config
    model_dim:int
    transformer_depth: int
    vocab_length: int

