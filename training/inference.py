from tokenizer.tokenizer import Tokenizer
from dataclasses import dataclass
from models.base_layers import softmax
import jax
from jax import numpy as jnp
from flax import linen as nn
from models.transformer import DeepSeekModel
from models.schemas import ModelConfig, MLA_config, MOE_FFN_config, Activation, RouterType
from training.training_utils import load_checkpoint
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm


def generate(model, params, tokenizer, text, key, max_new_tokens=100, top_p=50, top_k=100, temperature=0.7):
    # This logic matches the corrected inference loop
    tokens = tokenizer.inference_on_text(text)
    flattened_tokens = [tok.token_idx for word in tokens for tok in word]
    flattened_tokens_arr = jnp.array(flattened_tokens, dtype=jnp.int32)
    
    dummy_input = jnp.zeros((1, 1024), dtype=jnp.int32)
    dummy_input = jax.lax.dynamic_update_slice(dummy_input, flattened_tokens_arr[None, :], (0,0))
    
    num_tokens = len(flattened_tokens)
    last_token = flattened_tokens[-1]
    
    model_pass = jax.jit(model.apply)
    # Ensure params are in the right format. logic in train.py passes single device params.
    # If params is just the params dict, wrap it:
    model_vars = params if 'params' in params else {'params': params}
    
    generated_tokens = list(flattened_tokens)
    
    pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False)
    while last_token != 257 and num_tokens < 1024 and len(generated_tokens) < len(flattened_tokens) + max_new_tokens:
        out = model_pass(model_vars, dummy_input, rngs={'gumbel': key})
        
        logits = out[0][num_tokens-1] / temperature
        logits = logits - logits.max()
        probs = softmax(logits)
        
        # Pure JAX sorting - avoid Python loop for GPU efficiency
        sorted_indices = jnp.argsort(probs)[::-1]  # Descending order
        sorted_probs = probs[sorted_indices]
        
        top_k_indices = sorted_indices[:top_k]
        top_k_probs = sorted_probs[:top_k]
        
        cum_sum_prob = jnp.cumsum(top_k_probs)
        last_idx = jnp.searchsorted(cum_sum_prob > top_p, True)
        top_p_probs = top_k_probs[:last_idx]
        top_p_indices = top_k_indices[:last_idx]
        
        # Sampling with normalization
        key, subkey = jax.random.split(key)
        p_normalized = top_p_probs / jnp.sum(top_p_probs)
        selected_idx = jax.random.choice(subkey, jnp.arange(len(top_p_probs)), p=p_normalized)
        sampled_token = top_p_indices[selected_idx]

        
        # In-place update for next step
        dummy_input = dummy_input.at[0, num_tokens].set(sampled_token)
        
        sampled_token_int = int(sampled_token)
        generated_tokens.append(sampled_token_int)
        num_tokens += 1
        last_token = sampled_token_int
        pbar.update(1)
    pbar.close()

        
    try:
        decoded_text = ""
        for t in generated_tokens:
             token_obj = tokenizer.token_registery.get_token(int(t))
             if token_obj:
                 decoded_text += token_obj.byte_arr.decode('utf-8', errors='replace')
    except Exception:
        pass
        
    return decoded_text

@dataclass
class Inference:
    model_config: ModelConfig
    ckpt_path: str = None
    tokenizer_path: str = None
    
    def __post_init__(self):
        self.tokenizer = None
        self.model = DeepSeekModel(self.model_config)
        self.variables = None
        self._jitted_model = None  # Cache for JIT-compiled model
        
        # Load from checkpoint if path provided
        if self.ckpt_path:
            self.variables = self.load_model()
        if self.tokenizer_path:
            self.tokenizer = Tokenizer.load_for_inference(self.tokenizer_path)

    def load_model(self):
        if not os.path.exists(self.ckpt_path):
             raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")
        
        print(f"Loading checkpoint from {self.ckpt_path}")
        variables, _, _ = load_checkpoint(self.ckpt_path)
        variables = jax.device_put(variables)
        return variables

    def set_variables(self, variables):
        """Update model variables at runtime (e.g., during training)."""
        self.variables = variables
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer at runtime."""
        self.tokenizer = tokenizer

    def _get_jitted_model(self):
        """Get or create the JIT-compiled model (cached)."""
        if self._jitted_model is None:
            self._jitted_model = jax.jit(self.model.apply)
        return self._jitted_model

    def generate(self, text, key, max_new_tokens=100, top_p=50, top_k=100, temperature=0.7):
        """Generate text using cached JIT-compiled model."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        if self.variables is None:
            raise ValueError("Variables not set. Call set_variables() or load from checkpoint.")
        
        tokens = self.tokenizer.inference_on_text(text)
        flattened_tokens = [tok.token_idx for word in tokens for tok in word]
        flattened_tokens_arr = jnp.array(flattened_tokens, dtype=jnp.int32)
        
        dummy_input = jnp.zeros((1, 1024), dtype=jnp.int32)
        dummy_input = jax.lax.dynamic_update_slice(dummy_input, flattened_tokens_arr[None, :], (0,0))
        
        num_tokens = len(flattened_tokens)
        last_token = flattened_tokens[-1]
        
        model_pass = self._get_jitted_model()
        # Ensure params are in the right format
        model_vars = self.variables if 'params' in self.variables else {'params': self.variables}
        
        generated_tokens = list(flattened_tokens)
        
        pbar = tqdm(total=max_new_tokens, desc="Generating", leave=False)
        while last_token != 257 and num_tokens < 1024 and len(generated_tokens) < len(flattened_tokens) + max_new_tokens:
            out = model_pass(model_vars, dummy_input, rngs={'gumbel': key})
            
            logits = out[0][num_tokens-1] / temperature
            logits = logits - logits.max()
            probs = softmax(logits)
            
            # Pure JAX sorting - avoid Python loop for GPU efficiency
            sorted_indices = jnp.argsort(probs)[::-1]  # Descending order
            sorted_probs = probs[sorted_indices]
            
            top_k_indices = sorted_indices[:top_k]
            top_k_probs = sorted_probs[:top_k]
            
            cum_sum_prob = jnp.cumsum(top_k_probs)
            last_idx = jnp.searchsorted(cum_sum_prob > top_p, True)
            top_p_probs = top_k_probs[:last_idx]
            top_p_indices = top_k_indices[:last_idx]
            
            # Sampling with normalization
            key, subkey = jax.random.split(key)
            p_normalized = top_p_probs / jnp.sum(top_p_probs)
            selected_idx = jax.random.choice(subkey, jnp.arange(len(top_p_probs)), p=p_normalized)
            sampled_token = top_p_indices[selected_idx]

            
            # In-place update for next step
            dummy_input = dummy_input.at[0, num_tokens].set(sampled_token)
            
            sampled_token_int = int(sampled_token)
            generated_tokens.append(sampled_token_int)
            num_tokens += 1
            last_token = sampled_token_int
            pbar.update(1)
        pbar.close()
        
        try:
            decoded_text = ""
            for t in generated_tokens:
                 token_obj = self.tokenizer.token_registery.get_token(int(t))
                 if token_obj:
                     decoded_text += token_obj.byte_arr.decode('utf-8', errors='replace')
        except Exception:
            decoded_text = ""
            
        return decoded_text

    def inference(self, text, top_p=50, top_k=100, temprature=30, totoal_max_tokens=1024):
        """Legacy interface for standalone inference."""
        key = jax.random.PRNGKey(42)
        generated_text = self.generate(
            text=text,
            key=key,
            max_new_tokens=totoal_max_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temprature
        )
        print(generated_text)
        return generated_text


if __name__ == '__main__':
    # Configuration
    moe_mla_config = MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )
    
    moe_ffn_config = MOE_FFN_config(
        num_shared_experts=4,
        num_routing_experts=2,
        num_selected_experts=2,
        activation=Activation.LGELU.value,
        router_type=RouterType.LEARNED
    )
    
    moe_model_config = ModelConfig(
        mla_config=moe_mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=512,
        transformer_depth=28,
        vocab_length=32_000
    )

    from training.training_utils import find_latest_checkpoint
    
    ckpt_path = "/data3/vasu/projects/LMs-scratch-assignment1/checkpoints/learned_gelu/checkpoint_100000.pkl"
    # ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if not ckpt_path:
        root_ckpt_dir = "/data3/vasu/projects/LMs-scratch-assignment1/checkpoints"
        ckpt_path = find_latest_checkpoint(root_ckpt_dir)
        
    tokenizer_path = '/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl'

    if ckpt_path:
        print(f"Using checkpoint: {ckpt_path}")
        inference_engine = Inference(
            model_config=moe_model_config,
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path
        )

        test_text = "Once upon a time"
        inference_engine.inference(test_text)
    else:
        print("No checkpoint found.")
