"""
Inference test script for loading trained checkpoints and generating text.

This script demonstrates how to:
1. Load a trained model checkpoint
2. Load the tokenizer
3. Generate text from a prompt
"""

import jax
from jax import numpy as jnp
from models.transformer import DeepSeekModel
from models.schemas import ModelConfig, MLA_config, MOE_FFN_config, Activation, RouterType
from tokenizer.tokenizer import Tokenizer
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse


def load_checkpoint(checkpoint_path: str):
    """Load a model checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data


def generate_text(model, params, tokenizer, prompt="Once upon a time", max_len=100, temperature=1.0, top_k=50, key=None):
    """
    Generate text using the trained model.
    
    Args:
        model: The DeepSeekModel instance
        params: Model parameters from checkpoint
        tokenizer: Tokenizer instance
        prompt: Starting text prompt
        max_len: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        key: JAX random key
        
    Returns:
        Generated text string
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Tokenize the prompt
    tokens = tokenizer.inference_on_text(prompt, ret_type='int')
    # Flatten tokens if nested
    tokens = [t for sub in tokens for t in sub]
    input_ids = jnp.array([tokens])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Initial tokens: {tokens[:10]}... (showing first 10)")
    print(f"\nGenerating {max_len} tokens...\n")
    
    generated_tokens = []
    
    for i in tqdm(range(max_len), desc="Generating"):
        key, subkey = jax.random.split(key)
        
        # Get logits from model
        logits = model.apply({'params': params}, input_ids, rngs={'gumbel': subkey})
        last_logits = logits[:, -1, :]  # [B, V]
        
        # Apply temperature
        if temperature != 1.0:
            last_logits = last_logits / temperature
        
        # Top-k sampling
        if top_k > 0:
            # Get top-k values and indices
            top_k_values, top_k_indices = jax.lax.top_k(last_logits[0], top_k)
            
            # Sample from top-k
            key, sample_key = jax.random.split(key)
            # Apply softmax to get probabilities
            probs = jax.nn.softmax(top_k_values)
            # Sample from the distribution
            selected_idx = jax.random.categorical(sample_key, jnp.log(probs))
            next_token = top_k_indices[selected_idx]
        else:
            # Greedy decoding (argmax)
            next_token = jnp.argmax(last_logits, axis=-1)[0]
        
        # Add to sequence
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        generated_tokens.append(int(next_token))
        
        # Optional: break on end-of-text token (if you have one defined)
        # if next_token == 0:
        #     break
    
    # Convert tokens back to text
    all_tokens = input_ids[0].tolist()
    try:
        decoded_bytes = b''.join([
            tokenizer.token_registery.get_token(t).byte_arr 
            for t in all_tokens
        ])
        decoded_text = decoded_bytes.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Warning: Error during decoding: {e}")
        decoded_text = f"[Decoding error: {e}]"
    
    return decoded_text


def main():
    parser = argparse.ArgumentParser(description='Test inference with a trained checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_80000.pkl',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, 
                        default='tokenizer/trained/owt_train/final_0032000_inference.pkl',
                        help='Path to tokenizer checkpoint')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt to generate from')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter (0 for greedy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INFERENCE TEST")
    print("=" * 80)
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer from: {args.tokenizer}")
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
        return
    tokenizer = Tokenizer.load_for_inference(tokenizer_path)
    print(f"   Vocabulary size: {tokenizer.token_registery.num_tokens}")
    
    # Load checkpoint
    print(f"\n2. Loading checkpoint from: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    checkpoint_data = load_checkpoint(args.checkpoint)
    params = checkpoint_data['variables']['params']
    step = checkpoint_data.get('step', 'unknown')
    print(f"   Loaded checkpoint from step: {step}")
    
    # Create model config (should match the config used during training)
    print("\n3. Creating model configuration")
    mla_config = MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )
    
    moe_ffn_config = MOE_FFN_config(
        num_shared_experts=2,
        num_routing_experts=3,  # Match training config
        num_selected_experts=2,
        activation=Activation.RELU,
        router_type=RouterType.LEARNED
    )
    
    model_config = ModelConfig(   
        mla_config=mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=512,
        transformer_depth=10,
        vocab_length=32_000
    )
    
    # Create model
    print("   Model configuration:")
    print(f"   - Model dim: {model_config.model_dim}")
    print(f"   - Transformer depth: {model_config.transformer_depth}")
    print(f"   - Vocab size: {model_config.vocab_length}")
    print(f"   - MLA heads: {model_config.mla_config.num_heads}")
    print(f"   - MOE experts: {model_config.moe_ffn_config.num_routing_experts} routing + {model_config.moe_ffn_config.num_shared_experts} shared")
    
    model = DeepSeekModel(model_config=model_config)
    
    # Generate text
    print("\n4. Generating text")
    key = jax.random.PRNGKey(args.seed)
    
    generated_text = generate_text(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        key=key
    )
    
    print("\n" + "=" * 80)
    print("GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    # Try a few different prompts
    print("\n\nTrying additional prompts...\n")
    
    test_prompts = [
        "The king said",
        "In the forest",
        "She walked",
    ]
    
    for test_prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{test_prompt}'")
        print('='*60)
        key, subkey = jax.random.split(key)
        result = generate_text(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_len=50,  # Shorter for multiple prompts
            temperature=args.temperature,
            top_k=args.top_k,
            key=subkey
        )
        print(result)


if __name__ == '__main__':
    main()
