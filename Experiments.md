# List of Experiments
## Overfitting
### 100 Lines, around 3k tokens overfitting
#### Attempts
### Configuration

#### Model Architecture (MLA & MOE)
| Component | Parameter | Value |
| :--- | :--- | :--- |
| **MLA** | Latent Dim Q | 8 |
| | Latent Dim KV | 16 |
| | Content Dim | 512 |
| | Positional Dim | 128 |
| | Heads | 8 |
| **MOE FFN** | Shared Experts | 2 |
| | Routing Experts | 3 |
| | Selected Experts | 2 (Top-2 of 6) |
| | Activation | GELU |
| | Router Type | Learned |
| **General** | Model Dim | 512 |
| | Depth | 12 |
| | Vocab Size | 32,000 |

#### Training Settings
| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam |
| **Learning Rate** | 1e-5 (Lower for MOE) |
| **Seq Length** | 1024 |

### Run Logs & Observations
- Common Obvservation 

#### 1. Initial Attempt
- **Run**: [4zldjx0o](https://wandb.ai/vhaatever/LM-Training-Scratch/runs/4zldjx0o?nw=nwuservasuswaroop10)
- **Observations**: 
    - Greedy decoding is **degenerate**.
    - Loss and gradients **saturate** quickly.

#### 2. GELU Activation
- **Run**: [c3ylzcqq](https://wandb.ai/vhaatever/LM-Training-Scratch/runs/c3ylzcqq?nw=nwuservasuswaroop10)
- **Observations**: 
    - **Faster loss drop**.
    - **Higher Grad Norm** (likely indicating a less degenerate network being learned).

#### 3. Learned GELU Activations
##### 3.1
- The GELU paper claims that they model the activation as a product of x and a RV which is Bernoulli(Normal()). Whose expected value comes to be x*CDF(x). I just added learnable affine parameters before each RELU. Basically, choosing to cutoff values below x seemed arbitrary. Affine will just move zero point.
- **Run** : []
##### 3.2
- Adding the RMS norm before the final FFN layer
- **Run** : []
- **Observation**
- - Grad norms became very stable
- - The losses were also much lower 
##### 3.3
- Added truncated initialization, removed the bias terms in FFN
- **Run** : []
- **Observation**
#### 4. Switched to Minibatch overfitinh
##### Modifiactions
###### Exp 1
1. Only minibatch overfiting
2. Removed the MOE-gumbel head
###### Exp 2
1. Reduced the seq_len and increased depth, increased the LR
2. The loss came nice like expected
TODO: Ablate if it was depth or the seq_len. My gut says it was the seq_len, given a model_dim.
3. Post ablation, the issue was with learning rate only. Need to add a LR scheduler
BUGS: FFN output was undergoing RMS norm, RMS norm impplementation was non standard. Made it better and tested
4. Figured out it was code bugs, started triaging things
# 1. Data loading bugs
    * Started logging the training data at step t=0
    * Noticed wrong slicing of data for input and target
# 2. Attention
    * Was using matmuls, and ended up attending the head_dim and not the seq_len (fixed that)
    * Fixed the causal masking in LLM training
# 3. Added LR scheduling
# 4. Added Bfloat 16 training (Optimizer state in still in float32)
# 5. ABlated different model_dim, num_layers, etc. 
Working runs: https://wandb.ai/vhaatever/LM-Training-Scratch/runs/ddv99ntf? Model_dim 1024
        runs: https://wandb.ai/vhaatever/LM-Training-Scratch/runs/moliqvob 



# TODO List
[v] Add smarter innitialization
[v] Remove bias term in FFN
[v] In mixed precision, make sure to upcast properly
[] Benchmark against Swiglu
[v] Implement softmax 
[v] Implement masking
[v] Add a lr scheduler
[] Implement SWIGLU
[] Implement Canonical ROPE
[] Correct the MOE implementation
[] Implement scaling laws within a budget and automate it
---

### Strategy & Future Steps
> [!TIP]
> **Current Focus**: Adding functional tests to the codebase.
> 
> **Fallback Plan**: If progress stalls, I will simplify the architecture by:
> 1. Reducing **Transformer Depth** to 1.
> 2. Overfitting on a **single sequence length**.
> 3. Observing the **Logits Distribution** over time to diagnose the bottleneck.


