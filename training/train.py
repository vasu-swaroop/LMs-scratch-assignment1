from training.optimizer import AdamOptimizer
import jax
from jax import numpy as jnp
from models.transformer import DeepSeekModel
from models.schemas import ModelConfig, Activation
from dataclasses import dataclass
from jaxtyping import PRNGKeyArray, Float, Int, Array
import pickle
from flax import linen as nn

def cross_entropy_loss(pred:Float[Array, 'B S D'], target: Int[Float, 'B S']):
    pred_vector=nn.one_hot(target, pred.shape[-1])
    logits=jnp.log(pred)
    out=(pred_vector*logits).sum()
    return out

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
    
    def train_step(self, model, variables,optimizer_states,  input_data, out_data):

        @jax.jit
        def get_model_grads(model_vars, input_data, target_data):
            out=model.apply(variables, input_data)  
            loss=cross_entropy_loss(out, input_data)
            return loss   

        get_grads=jax.value_and_grad(get_model_grads)
        loss, grads=get_grads(variables, input_data, input_data)

        # import pdb; pdb.set_trace()
        variables, optimizer_states= AdamOptimizer().step(variables, grads, optimizer_states)

        #TODO: log grads

    
    def train(self):
        input_data= jnp.asarray(range(10000))
        # input_data= jnp.expand_dims(input_data, axis=0)
        input_data=jnp.stack([input_data]*1, axis=0)

        model=DeepSeekModel(self.model_config)
        variables=model.init(self.training_config.prng_key, input_data)

        optimizer_states=AdamOptimizer().init(variables)
        # import pdb; pdb.set_trace()
        self.train_step(model, variables,optimizer_states,  input_data, input_data)

        pass

if __name__=='__main__':
    # Test training configuration using values from transformer.py
    test_model_config = ModelConfig(
        latent_dim=8,
        hidden_dim=128,
        num_heads=6,
        model_dim=256,
        activation=Activation.RELU,
        transformer_depth=100,
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
    
    # Initialize training
    trainer = Training(test_train_settings, test_model_config)
    print("Test config created successfully!")
    print(f"Model config: {test_model_config}")
    print(f"Training settings: {test_train_settings}")
    trainer.train()