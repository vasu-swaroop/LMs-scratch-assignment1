import jax
import jax.numpy as jnp
from training.optimizer import AdamOptimizer

# Test if optimizer actually updates parameters
print("Testing AdamOptimizer...")

# Create simple params
params = {'weight': jnp.array([1.0, 2.0, 3.0])}
optimizer = AdamOptimizer(learning_rate=0.01)

# Initialize optimizer state
opt_state = optimizer.init(params)
print(f"Initial params: {params}")
print(f"Initial opt_state keys: {opt_state.keys()}")

# Create fake gradients (all ones, pushing weights down)
grads = {'weight': jnp.array([1.0, 1.0, 1.0])}

# Apply one optimizer step
new_params, new_opt_state = optimizer.step(params, grads, opt_state)

print(f"After 1 step with grads=[1,1,1]:")
print(f"  New params: {new_params}")
print(f"  Param change: {new_params['weight'] - params['weight']}")

# The params should have decreased (gradient descent)
assert jnp.all(new_params['weight'] < params['weight']), "Optimizer not updating params correctly!"

print("\n✓ Optimizer is working correctly!")
