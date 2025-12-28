"""
Utility functions for training including checkpoint management.
"""

import os
import glob
import re
import pickle
import jax
from jaxtyping import Float, Array

def save_checkpoint(variables, optimizer_states, step, checkpoint_dir):
    """
    Save a training checkpoint.
    
    Args:
        variables: Model variables (replicated across devices)
        optimizer_states: Optimizer states (replicated across devices)
        step: Current training step
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    checkpoint_data = {
        "variables": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], variables)),
        "optimizer_states": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], optimizer_states)),
        "step": step
    }
    with open(ckpt_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Saved checkpoint to {ckpt_path}")


def load_checkpoint(ckpt_path):
    """
    Load a training checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        Tuple of (variables, optimizer_states, step)
    """
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    return data["variables"], data["optimizer_states"], data["step"]


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    ckpt_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pkl")
    ckpts = glob.glob(ckpt_pattern)
    if ckpts:
        # Extract step number and find latest
        latest_ckpt = max(ckpts, key=lambda x: int(re.search(r'checkpoint_(\d+).pkl', x).group(1)))
        return latest_ckpt
    return None


def resume_from_checkpoint(checkpoint_dir):
    """
    Resume training from the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Tuple of (variables, optimizer_states, start_step) or (None, None, 0) if no checkpoint found
    """
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        vars_loaded, opt_loaded, step_loaded = load_checkpoint(latest_ckpt)
        
        # Replicate loaded variables across devices
        variables = {'params': jax.device_put_replicated(vars_loaded, jax.devices())}
        optimizer_states = jax.device_put_replicated(opt_loaded, jax.devices())
        
        print(f"Loaded step: {step_loaded}")
        return variables, optimizer_states, step_loaded
    else:
        print("No checkpoints found to resume from.")
        return None, None, 0

