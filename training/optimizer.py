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

    def init(self, params):
        return {
            "momentum": jax.tree_util.tree_map(jnp.zeros_like, params),
            "momentum_sq": jax.tree_util.tree_map(jnp.zeros_like, params),
            "step": jnp.array(1),  # JAX scalar
        }

    @partial(jax.jit, static_argnums=0)
    def step(self, params, grads, state, learning_rate):
        """
        params, grads, momentum, momentum_sq have identical PyTree structure
        learning_rate is a JAX scalar (replicated if pmapped)
        """

        step = state["step"]
        b1, b2 = self.beta_1, self.beta_2
        lr = learning_rate

        def update(p, g, m, v):
            # First and second moments
            m_new = b1 * m + (1.0 - b1) * g
            v_new = b2 * v + (1.0 - b2) * (g ** 2)

            # Bias correction
            m_hat = m_new / (1.0 - b1 ** step)
            v_hat = v_new / (1.0 - b2 ** step)

            # Parameter update
            p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

            return p_new, m_new, v_new

        mapped = jax.tree_util.tree_map(
            update,
            params,
            grads,
            state["momentum"],
            state["momentum_sq"],
        )
        
        outer_treedef = jax.tree_util.tree_structure(params)
        inner_treedef = jax.tree_util.tree_structure((0, 0, 0))
        new_params, new_momentum, new_momentum_sq = jax.tree_util.tree_transpose(outer_treedef, inner_treedef, mapped)

        new_state = {
            "momentum": new_momentum,
            "momentum_sq": new_momentum_sq,
            "step": step + 1,
        }

        return new_params, new_state

