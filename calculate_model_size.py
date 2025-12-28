"""
Calculate the number of parameters in the model based on the configuration.
"""

# Model Configuration
vocab_size = 32_000
model_dim = 512
transformer_depth = 12
num_heads = 8

# MLA Configuration
latent_dim_q = 8
latent_dim_kv = 16
dim_content = 512
dim_pos = 128

# MOE Configuration
num_shared_experts = 2
num_routing_experts = 3
num_selected_experts = 2  # Not used for param calculation

def calculate_mla_params():
    """Calculate parameters in one MLA layer"""
    # Q_d: model_dim -> latent_dim_q
    Q_d = model_dim * latent_dim_q + latent_dim_q
    
    # KV_d: model_dim -> latent_dim_kv
    KV_d = model_dim * latent_dim_kv + latent_dim_kv
    
    # Q_u_c: latent_dim_q -> dim_content * num_heads
    Q_u_c = latent_dim_q * (dim_content * num_heads) + (dim_content * num_heads)
    
    # K_u_c: latent_dim_kv -> dim_content
    K_u_c = latent_dim_kv * dim_content + dim_content
    
    # Q_u_p: latent_dim_q -> dim_pos * num_heads
    Q_u_p = latent_dim_q * (dim_pos * num_heads) + (dim_pos * num_heads)
    
    # K_u_p: latent_dim_kv -> dim_pos
    K_u_p = latent_dim_kv * dim_pos + dim_pos
    
    # V_U: latent_dim_kv -> dim_content
    V_U = latent_dim_kv * dim_content + dim_content
    
    # up_proj: (dim_content + dim_pos) -> model_dim
    hidden_dim = dim_content + dim_pos
    up_proj = hidden_dim * model_dim + model_dim
    
    total_mla = Q_d + KV_d + Q_u_c + K_u_c + Q_u_p + K_u_p + V_U + up_proj
    
    print(f"  MLA Layer Parameters Breakdown:")
    print(f"    Q_d: {Q_d:,}")
    print(f"    KV_d: {KV_d:,}")
    print(f"    Q_u_c: {Q_u_c:,}")
    print(f"    K_u_c: {K_u_c:,}")
    print(f"    Q_u_p: {Q_u_p:,}")
    print(f"    K_u_p: {K_u_p:,}")
    print(f"    V_U: {V_U:,}")
    print(f"    up_proj: {up_proj:,}")
    print(f"  Total MLA: {total_mla:,}")
    
    return total_mla

def calculate_ffn_params(input_dim, hidden_mult=4):
    """Calculate parameters in one FFN"""
    hidden_dim = input_dim * hidden_mult
    # Layer 1: input_dim -> hidden_dim
    layer1 = input_dim * hidden_dim + hidden_dim
    # Layer 2: hidden_dim -> input_dim
    layer2 = hidden_dim * input_dim + input_dim
    return layer1 + layer2

def calculate_moe_params():
    """Calculate parameters in one MOE layer"""
    # Shared experts: num_shared_experts FFNs
    shared_expert_params = num_shared_experts * calculate_ffn_params(model_dim)
    
    # Routing experts: num_routing_experts FFNs
    routing_expert_params = num_routing_experts * calculate_ffn_params(model_dim)
    
    # Router: model_dim -> num_routing_experts
    router_params = calculate_ffn_params(model_dim, hidden_mult=1)  # No hidden layer expansion in router
    # Actually the router goes: model_dim -> num_routing_experts
    router_params = model_dim * num_routing_experts + num_routing_experts
    
    total_moe = shared_expert_params + routing_expert_params + router_params
    
    print(f"  MOE Layer Parameters Breakdown:")
    print(f"    Shared experts ({num_shared_experts}x): {shared_expert_params:,}")
    print(f"    Routing experts ({num_routing_experts}x): {routing_expert_params:,}")
    print(f"    Router: {router_params:,}")
    print(f"  Total MOE: {total_moe:,}")
    
    return total_moe

def calculate_rms_norm_params():
    """RMSNorm has model_dim parameters (just scaling)"""
    return model_dim

def calculate_transformer_block_params():
    """Calculate parameters in one transformer block"""
    mla_params = calculate_mla_params()
    moe_params = calculate_moe_params()
    
    # 2 RMSNorm layers per block
    rms_norm_params = 2 * calculate_rms_norm_params()
    
    total_block = mla_params + moe_params + rms_norm_params
    
    print(f"  RMSNorm (2x): {rms_norm_params:,}")
    print(f"  Total per Transformer Block: {total_block:,}")
    
    return total_block

def calculate_total_params():
    """Calculate total model parameters"""
    print("="*80)
    print("MODEL PARAMETER CALCULATION")
    print("="*80)
    
    # Embedding layer
    embedding_params = vocab_size * model_dim
    print(f"\n1. Embedding Layer: {embedding_params:,}")
    print(f"   ({vocab_size:,} vocab × {model_dim} dim)")
    
    # Transformer blocks
    print(f"\n2. Transformer Block (per layer):")
    block_params = calculate_transformer_block_params()
    
    total_transformer_params = block_params * transformer_depth
    print(f"\n3. All Transformer Blocks ({transformer_depth} layers): {total_transformer_params:,}")
    
    # Output projection (final dense layer)
    output_params = model_dim * vocab_size + vocab_size
    print(f"\n4. Output Projection: {output_params:,}")
    print(f"   ({model_dim} × {vocab_size:,} + {vocab_size:,} bias)")
    
    # Total
    total_params = embedding_params + total_transformer_params + output_params
    
    print(f"\n{'='*80}")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"{'='*80}")
    
    # Calculate model size
    bytes_per_param = 4  # float32
    bytes_per_param_bf16 = 2  # bfloat16
    
    size_fp32_mb = (total_params * bytes_per_param) / (1024 * 1024)
    size_fp32_gb = size_fp32_mb / 1024
    
    size_bf16_mb = (total_params * bytes_per_param_bf16) / (1024 * 1024)
    size_bf16_gb = size_bf16_mb / 1024
    
    print(f"\nMODEL SIZE:")
    print(f"  FP32 (float32):    {size_fp32_mb:.2f} MB ({size_fp32_gb:.3f} GB)")
    print(f"  BF16 (bfloat16):   {size_bf16_mb:.2f} MB ({size_bf16_gb:.3f} GB)")
    print(f"\nParameters in millions: {total_params / 1_000_000:.2f}M")
    print("="*80)
    
    return total_params, size_fp32_mb, size_bf16_mb

if __name__ == "__main__":
    calculate_total_params()
