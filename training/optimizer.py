from jax import numpy as jnp
from dataclasses import dataclass
import jax
from functools import partial


def _init_dict( reference, inp):
    if type(inp)== dict:
        for key, values in inp.items():
            reference[key]={}
            _init_dict(reference, values)
    else:
        try:
            refernce=jnp.zeros_like(inp, dtype=float)
        except:
            import pdb; pdb.set_trace()
@dataclass(frozen=True)
class AdamOptimizer:
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    learning_rate: float = 1e-3

    def init(self, params):
        return {
            "momentum": jax.tree_util.tree_map(jnp.zeros_like, params),
            "momentum_sq": jax.tree_util.tree_map(jnp.zeros_like, params),
        }

    @partial(jax.jit, static_argnums=0)
    def step(self, params, grads, state):
        """
        params, grads, state are pytrees with identical structure
        """

        def update(p, g, m, v):
            m = self.beta_1 * m + (1.0 - self.beta_1) * g
            v = self.beta_2 * v + (1.0 - self.beta_2) * (g ** 2)
            p = p - self.learning_rate * m / (jnp.sqrt(v) + self.epsilon)
            return p, m, v

        out = jax.tree_util.tree_map(
            update,
            params,
            grads,
            state["momentum"],
            state["momentum_sq"],
        )

        params = jax.tree_util.tree_map(lambda x: x[0], out)
        momentum = jax.tree_util.tree_map(lambda x: x[1], out)
        momentum_sq = jax.tree_util.tree_map(lambda x: x[2], out)

        new_state = {
            "momentum": momentum,
            "momentum_sq": momentum_sq,
        }

        return params, new_state
