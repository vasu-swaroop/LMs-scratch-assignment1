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


# TODO List
[] Add smarter innitialization
[] Remove bias term in FFN
[] In mixed precision, make sure to upcast properly
[] Benchmark against Swiglu
[] Implement softmax 
[] Implement masking

---

### Strategy & Future Steps
> [!TIP]
> **Current Focus**: Adding functional tests to the codebase.
> 
> **Fallback Plan**: If progress stalls, I will simplify the architecture by:
> 1. Reducing **Transformer Depth** to 1.
> 2. Overfitting on a **single sequence length**.
> 3. Observing the **Logits Distribution** over time to diagnose the bottleneck.


