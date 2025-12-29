from training.optimizer import AdamOptimizer
import jax
from jax import numpy as jnp
from models.transformer import DeepSeekModel
from dataclasses import dataclass
from jaxtyping import PRNGKeyArray, Float, Int, Array
import pickle
from models.schemas import MLA_config, MOE_FFN_config, RouterType, ModelConfig, Activation
from training.data import Data
from tokenizer.tokenizer import Tokenizer
from pathlib import Path
from tqdm import tqdm
from flax import linen as nn
import wandb
import os
import glob
import re
# from training.inference import generate
from training.training_utils import save_checkpoint, load_checkpoint, resume_from_checkpoint
jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_default_matmul_precision", "bfloat16")

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
    batch_size: int
    resume_ckpt: bool
    seq_len:int
    train_steps:int
    data_path: Path
    val_data_path: Path
    prng_key: PRNGKeyArray
    grad_accumulation: int
    num_gpus:int
    checkpoint_dir: str = "checkpoints/learned_gelu"
    save_every: int = 20000
    val_every: int = 5000
    val_batches: int = 1
    wandb_project: str = "LM-Training-Scratch"
    wandb_run_name: str = "CorrectRMSNorm_TinyStoriesOverfiting-100Lines-LGelu-RMSNormEnd_trunc_init_minibatch_noexpert_only_lr"
    use_wandb: bool = True
    
class Training():
    def __init__(self, training_settings:TrainSettings, model_settings:ModelConfig):
        self.training_config=training_settings
        self.model_config=model_settings


    def train_step(
        self,
        model,
        variables,
        optimizer_states,
        input_data,
        out_data,
        key,
        accum_grads,
        grad_step=True,
    ):
        def loss_fn(params, input_ids, targets, rng):
            logits = model.apply({'params': params}, input_ids, rngs={'gumbel': rng})
            ce_loss = cross_entropy_loss(logits, targets)
            return ce_loss, ce_loss  # Return loss and aux (ce_loss for logging)

        def step_fn(params, input_ids, targets, rng, current_grads, do_opt, opt_state):
            (loss, ce_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, input_ids, targets, rng)
            
            # Average loss and grads across devices
            loss = jax.lax.pmean(loss, axis_name='data')
            ce_loss = jax.lax.pmean(ce_loss, axis_name='data')
            grads = jax.lax.pmean(grads, axis_name='data')
            
            # Clip and accumulate
            grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
            new_grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, current_grads)
            
            # Calculate norm of accumulated grads BEFORE resetting
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(new_grads)))
            
            def perform_opt():
                optimizer = AdamOptimizer(learning_rate=self.training_config.lr)
                updated_params, updated_opt_state = optimizer.step(params, new_grads, opt_state)
                reset_grads = jax.tree_util.tree_map(jnp.zeros_like, new_grads)
                return updated_params, updated_opt_state, reset_grads

            def skip_opt():
                return params, opt_state, new_grads

            res_params, res_opt_state, res_grads = jax.lax.cond(do_opt, perform_opt, skip_opt)
            return res_params, res_opt_state, res_grads, loss, grad_norm, ce_loss

        #Otherwise everytime we call train_step, we would call rcompilation.
        if not hasattr(self, '_pmapped_train_step'):
            self._pmapped_train_step = jax.pmap(
                step_fn, 
                axis_name='data', 
                static_broadcasted_argnums=(5,)
            )

        num_devices = jax.local_device_count()
        keys = jax.random.split(key, num_devices)

        new_params, new_opt_state, new_grads, loss, grad_norm, ce_loss = self._pmapped_train_step(
            variables['params'],
            input_data,
            out_data,
            keys,
            accum_grads,
            grad_step,
            optimizer_states
        )

        return {'params': new_params}, loss, new_grads, new_opt_state, grad_norm, ce_loss


    def _val_step(self, model, variables, input_data, out_data, key):
        """Validation step runs on a single GPU for simplicity."""
        def loss_fn(params, input_data, target_data, key):
            out = model.apply({'params': params}, input_data, rngs={'gumbel': key})
            return cross_entropy_loss(out, target_data)

        # Use only the first device's parameters for validation
        params = jax.tree_util.tree_map(lambda x: x[0], variables['params'])
        return loss_fn(params, input_data, out_data, key)

    def resize_for_multinode(self, inp):
        num_gpus=self.training_config.num_gpus
        per_device_batch=self.training_config.batch_size//num_gpus
        inp = inp.reshape(
            num_gpus,
            per_device_batch,
            *inp.shape[1:]
        )
        return inp
    def train(self, dataset, val_dataset=None):
        # Initialize W&B
        if self.training_config.use_wandb:
            wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name,
                config=self.training_config.__dict__,
                # id='uf7n20vn',
                # resume="allow",
            )

        input_data, _ = next(dataset.data_loader())
        key = self.training_config.prng_key

        model = DeepSeekModel(self.model_config)
        variables = model.init({'params': key, 'gumbel': key}, input_data)
        variables = jax.device_put_replicated(
            variables, jax.devices()
        )

        # For generation
        tokenizer = Tokenizer.load_for_inference('/data3/vasu/projects/LMs-scratch-assignment1/tokenizer/trained/owt_train/final_0032000_inference.pkl')
        # Initialize grads with the same replicated structure as variables['params']
        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), variables['params'])

        grad_accum_step_size = self.training_config.grad_accumulation
        # Initialize optimizer states and replicate
        optimizer_states = AdamOptimizer().init(jax.tree_util.tree_map(lambda x: x[0], variables['params']))
        optimizer_states = jax.device_put_replicated(optimizer_states, jax.devices())
        
        start_step = 0
        if self.training_config.resume_ckpt:
            resumed_vars, resumed_opt, start_step = resume_from_checkpoint(self.training_config.checkpoint_dir)
            if resumed_vars is not None:
                variables = resumed_vars
                optimizer_states = resumed_opt

        data_gen = dataset.data_loader()
        val_gen = val_dataset.data_loader() if val_dataset else None
        
        pbar = tqdm(enumerate(data_gen, start=start_step), total=self.training_config.train_steps, initial=start_step)
        for idx, (input_data, output_data) in pbar:
            input_data, output_data = self.resize_for_multinode(input_data), self.resize_for_multinode(output_data)
            key, subkey = jax.random.split(key)
            
            update_weights = (idx + 1) % grad_accum_step_size == 0

            variables, loss, grads, optimizer_states, grad_norm, ce_loss = self.train_step(
                model, variables, optimizer_states, input_data, output_data, subkey, grads, grad_step=update_weights
            )

            loss_value = float(loss[0])
            grad_norm_value = float(grad_norm[0])
            ce_loss_value = float(ce_loss[0])

            pbar.set_description(f"loss={loss_value:.4f}, ce={ce_loss_value:.4f}, grad_norm={grad_norm_value:.2f}")
            
            # W&B Logging
            if update_weights and self.training_config.use_wandb:
                # Calculate useful metrics
                effective_grad_steps = (idx + 1) // grad_accum_step_size  # Number of actual optimizer updates
                tokens_seen = (idx + 1) * self.training_config.batch_size * self.training_config.seq_len
                effective_batch_size = self.training_config.batch_size * grad_accum_step_size
                
                wandb.log({
                    "train/loss": loss_value,
                    "train/ce_loss": ce_loss_value,
                    "train/grad_norm": grad_norm_value,
                    "step": idx,
                    "metrics/effective_grad_steps": effective_grad_steps,
                    "metrics/tokens_seen": tokens_seen,
                    "metrics/effective_batch_size": effective_batch_size,
                })

            # Periodic Checkpointing
            if (idx + 1) % self.training_config.save_every == 0:
                save_checkpoint(variables, optimizer_states, idx + 1, self.training_config.checkpoint_dir)

            # Periodic Validation / Generation
            if (idx + 1) % self.training_config.val_every == 0:
                gen_key, key = jax.random.split(key)
                
                # Efficient Validation Loss
                val_losses = []
                if val_gen:
                    for _ in range(self.training_config.val_batches):
                        try:
                            v_input, v_output = next(val_gen)
                        except StopIteration:
                            val_gen = val_dataset.data_loader()
                            v_input, v_output = next(val_gen)
                        
                        # Validation runs on single GPU - no need to resize
                        v_loss = self._val_step(model, variables, v_input, v_output, gen_key)
                        val_losses.append(float(v_loss))
                
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                # sample = generate(model, variables['params'], tokenizer, key=gen_key)
                
                print(f"\n[Step {idx+1}] Val Loss: {avg_val_loss:.4f}")
                if self.training_config.use_wandb:
                    wandb.log({
                        "val/loss": avg_val_loss,
                        # "val/generated_text": wandb.Html(sample), 
                        "step": idx+1
                    })

        if self.training_config.use_wandb:
            wandb.finish()



if __name__=='__main__':
    
    moe_mla_config = MLA_config(
        latent_dim_q=8,
        latent_dim_kv=16,
        dim_content=512,
        dim_pos=128,
        num_heads=8,
    )
    
    moe_ffn_config = MOE_FFN_config(
        num_shared_experts=2,      # Shared experts (general number = 2)
        num_routing_experts=2,      # Total routing experts available
        num_selected_experts=2,     # Top-k selection (selects top 2 out of 6)
        activation=Activation.LGELU.value,
        router_type=RouterType.LEARNED
    )
    
    moe_model_config = ModelConfig(
        mla_config=moe_mla_config,
        moe_ffn_config=moe_ffn_config,
        model_dim=512,
        transformer_depth=12,
        vocab_length=32_000
    )
    
    moe_train_settings = TrainSettings(
        optimizer='adam',
        lr=1e-5,                    # Lower learning rate for MOE models
        batch_size=2,              # Reduced for stability/OOM
        resume_ckpt=False,
        train_steps=1000000,
        seq_len=1024,
        prng_key=jax.random.PRNGKey(42),
        data_path=Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_tineystoruies_100_lines'),
        val_data_path=Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_tineystoruies_100_lines'),
        grad_accumulation=16,
        num_gpus=2,
        use_wandb=True,
        save_every=10000
    )

    train_dataset=Data(moe_train_settings.data_path,moe_train_settings.batch_size,moe_train_settings.train_steps, moe_train_settings.seq_len)
    val_dataset=Data(moe_train_settings.val_data_path,moe_train_settings.batch_size,moe_train_settings.train_steps, moe_train_settings.seq_len)
    trainer = Training(moe_train_settings, moe_model_config)
    trainer.train(train_dataset, val_dataset)