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

        def update_param(p, g, m, v):
            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * (g ** 2)
            p_new = p - self.learning_rate * m_new / (jnp.sqrt(v_new) + self.epsilon)
            return p_new
        
        def update_momentum(p, g, m, v):
            return self.beta_1 * m + (1.0 - self.beta_1) * g
        
        def update_momentum_sq(p, g, m, v):
            return self.beta_2 * v + (1.0 - self.beta_2) * (g ** 2)

        new_params = jax.tree_util.tree_map(
            update_param,
            params,
            grads,
            state["momentum"],
            state["momentum_sq"],
        )
        
        new_momentum = jax.tree_util.tree_map(
            update_momentum,
            params,
            grads,
            state["momentum"],
            state["momentum_sq"],
        )
        
        new_momentum_sq = jax.tree_util.tree_map(
            update_momentum_sq,
            params,
            grads,
            state["momentum"],
            state["momentum_sq"],
        )

        new_state = {
            "momentum": new_momentum,
            "momentum_sq": new_momentum_sq,
        }

        return new_params, new_state
