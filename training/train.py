from training.optimizer import AdamOptimizer
import jax
from jax import numpy as jnp
from models.transformer import DeepSeekModel
from dataclasses import dataclass
from jaxtyping import PRNGKeyArray, Float, Int, Array
import pickle
from models.schemas import MLA_config, MOE_FFN_config, RouterType, ModelConfig, Activation
from training.data import Data
from pathlib import Path
from tqdm import tqdm
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
    seq_len:int
    train_steps:int
    data_path: Path
    prng_key: PRNGKeyArray
    grad_accumulation: int
    num_gpus:int
    
class Training():
    def __init__(self, training_settings:TrainSettings, model_settings:ModelConfig):
        self.training_config=training_settings
        self.model_config=model_settings

    def load_model(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            data=pickle.load(ckpt_path)
    
    def train_step(
        self,
        model,
        variables,
        optimizer_states,
        input_data,
        out_data,
        key,
        orig_grads,
        grad_step=True,
    ):

        def loss_fn(params, input_data, target_data, key):
            out = model.apply(
                {'params': params},
                input_data,
                rngs={'gumbel': key}
            )
            loss = cross_entropy_loss(out, target_data)
            return loss

        def loss_and_grad_pmapped(params, input_data, target_data, key):
            loss, grads = jax.value_and_grad(loss_fn)(
                params, input_data, target_data, key
            )

            loss = jax.lax.pmean(loss, axis_name='data')
            grads = jax.lax.pmean(grads, axis_name='data')

            return loss, grads

        loss_and_grad_pmapped = jax.pmap(
            loss_and_grad_pmapped,
            axis_name='data'
        )

        # keys MUST be per-device
        keys = jax.random.split(key, jax.local_device_count())

        loss, grads = loss_and_grad_pmapped(
            variables['params'],
            input_data,
            out_data,
            keys
        )

        # safe to do non-collective ops outside
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, orig_grads)

        new_params, optimizer_states = AdamOptimizer().step(
            variables['params'],
            grads,
            optimizer_states
        )

        if grad_step:
            variables = {'params': new_params}

        return variables, loss, grads, optimizer_states

    def resize_for_multinode(self, inp):
        num_gpus=self.training_config.num_gpus
        per_device_batch=self.training_config.batch_size//num_gpus
        inp = inp.reshape(
            num_gpus,
            per_device_batch,
            *inp.shape[1:]
        )
        return inp
    def train(self, dataset):
        input_data, _ =next(dataset.data_loader())
        key = self.training_config.prng_key

        model = DeepSeekModel(self.model_config)
        variables = model.init({'params': key, 'gumbel': key}, input_data)
        variables = jax.device_put_replicated(
            variables, jax.devices()
        )


        grads=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), variables['params'])

        grad_accum_step_size=self.training_config.grad_accumulation
        prev_grad_norm=0
        optimizer_states = AdamOptimizer().init(variables['params'])
        
        pbar = tqdm(enumerate(dataset.data_loader()), total=self.training_config.train_steps)
        for idx, (input_data, output_data) in pbar:
            input_data, output_data=self.resize_for_multinode(input_data), self.resize_for_multinode(output_data)
            key, subkey = jax.random.split(key)
            if idx%grad_accum_step_size==0:
                grads=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), variables['params'])

            update_weights=idx+1%grad_accum_step_size==0

            variables, loss, grads, optimizer_states = self.train_step(
                model, variables, optimizer_states, input_data, output_data, subkey, grads, grad_step=update_weights
            )

            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
            loss_value = float(loss[0])
            grad_norm_value = float(grad_norm)

            pbar.set_description(f"loss={loss_value:.4f}, grad_norm={grad_norm_value:.2f}")

            if not update_weights:
                prev_grad_norm=grad_norm
            else:
                prev_grad_norm=0


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
        batch_size=8,              # Smaller batch size due to MOE complexity
        cur_steps=0,
        cur_epoch=0,
        resume_ckpt=False,
        train_steps=100000,
        seq_len=1000,
        prng_key=jax.random.PRNGKey(42),
        data_path=Path('/data3/vasu/projects/LMs-scratch-assignment1/train_data/overfiting_test_tineystoruies_train'),
        grad_accumulation=4,
        num_gpus=2
    )

    dataset=Data(moe_train_settings.data_path,moe_train_settings.batch_size,moe_train_settings.train_steps, moe_train_settings.seq_len)
    trainer = Training(moe_train_settings, moe_model_config)
    trainer.train(dataset)