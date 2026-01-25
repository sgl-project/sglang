"""
Fix for Issue #17680: MoE Tensor Parallelism Bug

This fix adds padding logic to handle weight dimensions that are not properly 
aligned when using tensor parallelism with MoE models.

File: python/sglang/srt/layers/moe/fused_moe_triton/layer.py
"""

# ============================================================================
# CHANGE 1: Add import at the top of the file (around line 36)
# ============================================================================

# BEFORE:
# from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
# from sglang.srt.utils import (
#     cpu_has_amx_support,
#     get_bool_env_var,
#     is_cpu,
#     is_flashinfer_available,
#     is_hip,
#     next_power_of_2,
#     round_up,
# )

# AFTER:
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.layers.utils import pad_or_narrow_weight  # ADD THIS LINE
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_flashinfer_available,
    is_hip,
    next_power_of_2,
    round_up,
)


# ============================================================================
# CHANGE 2: Fix _load_w2 method (around line 428-434)
# ============================================================================

def _load_w2_fixed(
    self,
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    is_bias: bool = False,
):
    """Load w2 weights for down projection.
    
    FIXED VERSION: Adds padding logic for weight dimensions that are not properly aligned.
    """
    if not isinstance(expert_data, torch.Tensor) or not isinstance(
        loaded_weight, torch.Tensor
    ):
        raise ValueError("expert_data and loaded_weight must be torch.Tensor")

    if (
        self.quant_config is not None
        and "modelopt" in self.quant_config.get_name()
        and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
    ):
        raise ValueError(
            f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
        )

    if shard_id != "w2":
        raise ValueError(f"shard_id must be 'w2', got {shard_id}")

    # Index the loaded weight for tp sharding.
    # down_proj: "RowParallel" so tp sharding on input_dim
    # Narrow parameter and load.
    if is_bias:
        # this expert_data is a bias, not weight,
        # for w2_weight_bias in TP, it does not need to be sharded
        shard_size = expert_data.shape[-1]
    else:
        # this parameter is a weight matrix
        # for w2 in TP, it shards the input_features, i.e., shard_dim=2
        shard_size = expert_data.shape[shard_dim]

    if _is_cpu:
        expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
            expert_data,
            loaded_weight,
            0,  # param_data_start
            shard_size * tp_rank,
            shard_dim,
            shard_size,
            not self.use_presharded_weights,
        )
    else:
        if not is_bias and not self.use_presharded_weights:
            if self.use_triton_kernels:
                loaded_weight = loaded_weight.transpose(-2, -1)
            
            # ================================================================
            # FIXED CODE: Add boundary check and padding logic
            # ================================================================
            # Padding for special case where weight dimension is not properly aligned
            start_idx = shard_size * tp_rank
            end_idx = start_idx + shard_size
            if end_idx > loaded_weight.shape[shard_dim]:
                loaded_weight = pad_or_narrow_weight(
                    loaded_weight, shard_dim, start_idx, shard_size
                )
            else:
                loaded_weight = loaded_weight.narrow(
                    shard_dim, start_idx, shard_size
                )
            # ================================================================
            # BEFORE (BROKEN CODE):
            # loaded_weight = loaded_weight.narrow(
            #     shard_dim, shard_size * tp_rank, shard_size
            # )
            # ================================================================

    # w2, down_proj: Load into only logical weight of w2.
    expert_data.copy_(loaded_weight)


# ============================================================================
# EXPLANATION
# ============================================================================
"""
The fix adds boundary checking before calling narrow():

1. Calculate start_idx and end_idx
2. Check if end_idx exceeds the dimension size
3. If yes, use pad_or_narrow_weight() to handle padding
4. If no, use narrow() as before

This handles cases where:
- Weight dimensions are not properly aligned (e.g., quantized models)
- TP rank 1 tries to access [8:16] but dimension is only 8
- Padding with zeros ensures the operation succeeds

The fix is consistent with how RowParallelLinear handles similar cases.
"""
