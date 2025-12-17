from enum import Enum
from flax import linen as nn

class Activation(Enum):
    RELU = nn.relu
    GELU = nn.gelu
    SWISH = nn.swish
    
    def __call__(self, x):
        return self.value(x)


