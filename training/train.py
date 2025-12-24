from training.optimizer import AdamOptimizer
import jax
from jax import numpy as jnp
from models.transformer import DeepSeekModel
from dataclasses import dataclass
from jaxtyping import PRNGKeyArray, Float, Int, Array
import pickle
from models.schemas import MLA_config, MOE_FFN_config, RouterType, ModelConfig, Activation

from flax import linen as nn

def cross_entropy_loss(logits: Float[Array, 'B S D'], target: Int[Float, 'B S']):
    """
    Numerically stable cross-entropy loss.
    Args:
        logits: Raw model outputs (not softmax), shape [B, S, D]
        target: Target token indices, shape [B, S]
    """
    # Use log_softmax for numerical stability instead of log(softmax(x))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_one_hot = jax.nn.one_hot(target, logits.shape[-1])
    loss = -jnp.sum(target_one_hot * log_probs) / (target.shape[0] * target.shape[1])
    return loss

@dataclass
class TrainSettings:
    optimizer: str
    lr: float
    num_epochs: int
    batch_size: int
    cur_steps: int
    cur_epoch: int
    resume_ckpt: bool
    prng_key: PRNGKeyArray
    
class Training():
    def __init__(self, training_settings:TrainSettings, model_settings:ModelConfig):
        self.training_config=training_settings
        self.model_config=model_settings

    def load_model(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            data=pickle.load(ckpt_path)
    
    def train_step(self, model, variables, optimizer_states, input_data, out_data, key):

        def get_model_grads(params, input_data, target_data):
            out = model.apply({'params': params}, input_data, rngs={'gumbel': key})  
            loss = cross_entropy_loss(out, target_data)
            return loss   

        get_grads = jax.value_and_grad(get_model_grads)
        loss, grads = get_grads(variables['params'], input_data, input_data)

        # Clip gradients to prevent explosion
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        # Update params via optimizer
        new_params, optimizer_states = AdamOptimizer().step(variables['params'], grads, optimizer_states)
        variables = {'params': new_params}

        return variables, loss, grads, optimizer_states
    
    def train(self):
        input_data = jnp.asarray(range(10000))
        # input_data = jnp.expand_dims(input_data, axis=0)
        input_data = jnp.stack([input_data]*1, axis=0)

        model = DeepSeekModel(self.model_config)
        key = self.training_config.prng_key

        variables = model.init({'params': key, 'gumbel': key}, input_data)

        optimizer_states = AdamOptimizer().init(variables['params'])
        for i in range(100):
            key, subkey = jax.random.split(key)
            variables, loss, grads, optimizer_states = self.train_step(
                model, variables, optimizer_states, input_data, input_data, subkey
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
            print(f"Step {i}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")


if __name__=='__main__':
    # Import required schemas
    from models.schemas import MLA_config, MOE_FFN_config, RouterType
    
    # Test training configuration using values from transformer.py
    test_mla_config = MLA_config(
        latent_dim_q=4,
        latent_dim_kv=8,
        dim_content=256,
        dim_pos=64,
        num_heads=6,
    )
    
    test_moe_config = MOE_FFN_config(
        num_shared_experts=1,
        num_routing_experts=4,
        num_selected_experts=2,
        expert_dim=512,
        activation=Activation.RELU,
        router_type=RouterType.LEARNED
    )
    
    test_model_config = ModelConfig(
        mla_config=test_mla_config,
        moe_ffn_config=test_moe_config,
        model_dim=256,
        transformer_depth=2,
        vocab_length=32_000
    )
    
    test_train_settings = TrainSettings(
        optimizer='adam',
        lr=3e-4,
        num_epochs=10,
        batch_size=32,
        cur_steps=0,
        cur_epoch=0,
        resume_ckpt=False,
        prng_key=jax.random.PRNGKey(42)
    )
    
    # Sample MOE (Mixture-of-Experts) configuration
    
    moe_mla_config = MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )
    
    moe_ffn_config = MOE_FFN_config(
        num_shared_experts=2,      # Shared experts (general number = 2)
        num_routing_experts=3,      # Total routing experts available
        num_selected_experts=2,     # Top-k selection (selects top 2 out of 6)
        expert_dim=512,           # Hidden dimension for expert FFNs
        activation=Activation.RELU,
        router_type=RouterType.LEARNED
    )
    
    moe_model_config = ModelConfig(
        mla_config=moe_mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=512,
        transformer_depth=10,
        vocab_length=32_000
    )
    
    moe_train_settings = TrainSettings(
        optimizer='adam',
        lr=1e-6,                    # Lower learning rate for MOE models
        num_epochs=20,
        batch_size=16,              # Smaller batch size due to MOE complexity
        cur_steps=0,
        cur_epoch=0,
        resume_ckpt=False,
        prng_key=jax.random.PRNGKey(42)
    )
    
    # Initialize training
    trainer = Training(test_train_settings, test_model_config)
    print("Test config created successfully!")
    print(f"Model config: {test_model_config}")
    print(f"Training settings: {test_train_settings}")
    
    print("\n" + "="*60)
    print("MOE Configuration created successfully!")
    print("="*60)
    print(f"\nMOE Model config: {moe_model_config}")
    print(f"\nMOE Training settings: {moe_train_settings}")
    print("\n" + "="*60)
    
    # Uncomment to train with MOE config instead:
    # moe_trainer = Training(moe_train_settings, moe_model_config)
    # moe_trainer.train()
    
    trainer.train()