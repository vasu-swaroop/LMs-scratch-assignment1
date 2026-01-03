from flax import linen as nn
from jaxtyping import Float, Array
from jax import numpy as jnp
class LearnedGelu(nn.Module):
    '''Want to experiment with implicit statistical learning of the Gausian the GELU approximates
    This would be like trying to shift the zero/clip point and the strength of non linearity.
    As sigma goes to zero, becomes more and more like Geli'''
    dtype : jnp.dtype = jnp.bfloat16
    @nn.compact
    def __call__(self, inp:Float[Array, '... D'])->Float[Array, '... D']:
        mean=self.param('mean', nn.initializers.constant(0),(),dtype=self.dtype)
        scale=self.param('scale', nn.initializers.constant(1),(),dtype=self.dtype)
        inp=(inp-mean)*scale
        return nn.gelu(inp).astype(self.dtype)


# from jax import numpy as np
# model=LearnedGelu()
# inp=np.ones((10,94,23))
# vars=model.init(jax.random.PRNGKey(42), inp)
# print(vars)
# out=model.apply(vars,inp)
# print((inp.shape, out[0][0]))