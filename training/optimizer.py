from jax import numpy as jnp
from dataclasses import dataclass
import jax
from functools import partial


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

    def update(self, p, g, m, v, learning_rate, step):
        lr=learning_rate
        # First and second moments
        b1, b2= self.beta_1, self.beta_2
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * (g ** 2)

        # Bias correction
        m_hat = m_new / (1.0 - b1 ** step)
        v_hat = v_new / (1.0 - b2 ** step)

        # Parameter update
        p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

        return p_new, m_new, v_new

    @partial(jax.jit, static_argnums=0)
    def step(self, params, grads, state, learning_rate):
        """
        params, grads, momentum, momentum_sq have identical PyTree structure
        learning_rate is a JAX scalar (replicated if pmapped)
        """

        step = state["step"]
        mapped = jax.tree_util.tree_map(
            partial(self.update, learning_rate=learning_rate, step=step),
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
    
@dataclass(frozen=True)
class MuonOptimizer(AdamOptimizer):
    mu:float= 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    def init(self, params):
        return {
            "muon":{"b":jax.tree_util.tree_map(jnp.zeros_like, params),
                },
            "adam":{"momentum": jax.tree_util.tree_map(jnp.zeros_like, params),
            "momentum_sq": jax.tree_util.tree_map(jnp.zeros_like, params),
                },
            "step": jnp.array(1), 
            }
    
    def newton_schulz(self, x, num_iters=5):
        """
        Newton-Schulz iteration for computing orthogonalized matrix.
        Approximates X @ (X^T @ X)^{-1/2} which orthogonalizes the columns of X.
        """
        a, b, c = (3.4445, -4.7750, 2.0315)  # Coefficients for cubic iteration
        
        # Transpose if necessary (shape is static, so Python if is ok)
        transpose = x.shape[0] < x.shape[1]
        if transpose:
            x = x.T
            
        # Normalize to improve convergence
        norm = jnp.linalg.norm(x)
        x = x / (norm + 1e-7)
        
        # Newton-Schulz iterations using fori_loop (avoids retracing from Python for)
        def ns_iteration(_, x):
            A = x @ x.T
            B = b * A + c * A @ A
            return a * x + B @ x
        
        x = jax.lax.fori_loop(0, num_iters, ns_iteration, x)
            
        if transpose:
            x = x.T
            
        return x
    
    def _muon_update(self, p, g, muon_b, adam_m, adam_v, learning_rate, step_count):
        """Muon update for 2D (matrix) parameters."""
        b_new = muon_b * self.mu + g
        o_new = self.newton_schulz(b_new)
        p_new = p - o_new * learning_rate
        return p_new, b_new, adam_m, adam_v
    
    def _adam_update(self, p, g, muon_b, adam_m, adam_v, learning_rate, step_count):
        """Adam update for 1D parameters."""
        p_new, m_new, v_new = self.update(p, g, adam_m, adam_v, learning_rate, step_count)
        return p_new, muon_b, m_new, v_new
    
    # NOTE: No @jax.jit here - this is called from inside pmap which handles compilation
    def step(self, params, grads, state, learning_rate):
        """
        Apply Muon optimizer to 2D parameters and Adam to 1D parameters.
        """
        step_count = state["step"]
        
        # Pre-compute which params are 2D (this is static/shape-based, not traced)
        is_2d = jax.tree_util.tree_map(lambda p: p.ndim == 2, params)
        
        # Apply updates based on dimensionality (static dispatch based on ndim)
        def update_param(p, g, muon_b, adam_m, adam_v, is_matrix):
            if is_matrix:  # This is a Python bool (static), not a traced value
                return self._muon_update(p, g, muon_b, adam_m, adam_v, learning_rate, step_count)
            else:
                return self._adam_update(p, g, muon_b, adam_m, adam_v, learning_rate, step_count)
        
        # Apply updates to all parameters
        mapped = jax.tree_util.tree_map(
            update_param,
            params,
            grads,
            state["muon"]["b"],
            state["adam"]["momentum"],
            state["adam"]["momentum_sq"],
            is_2d,
        )
        
        # Transpose to separate the results
        outer_treedef = jax.tree_util.tree_structure(params)
        inner_treedef = jax.tree_util.tree_structure((0, 0, 0, 0))
        new_params, new_muon_b, new_adam_m, new_adam_v = jax.tree_util.tree_transpose(
            outer_treedef, inner_treedef, mapped
        )
        
        new_state = {
            "muon": {"b": new_muon_b},
            "adam": {
                "momentum": new_adam_m,
                "momentum_sq": new_adam_v,
            },
            "step": step_count + 1,
        }
        
        return new_params, new_state


def test_muon():
    import jax
    import jax.numpy as jnp

    # Create a simple PyTree using a dictionary
    key = jax.random.PRNGKey(42)
    key_w, key_b = jax.random.split(key)

    simple_layer_params = {
        'weights': jax.random.normal(key_w, (3, 2)), # A JAX array
        'bias': jax.random.normal(key_b, (2,))      # Another JAX array
    }

    print("Simple PyTree (layer parameters):")
    print(simple_layer_params)
    
    opt=MuonOptimizer()
    states=opt.init(simple_layer_params)
    
    simple_layer_grads = {
        'weights': jax.random.normal(key_w, (3, 2)), # A JAX array
        'bias': jax.random.normal(key_b, (2,))      # Another JAX array
    }

    opt.step(simple_layer_params,simple_layer_grads, states, jnp.asarray(0.001))

test_muon()