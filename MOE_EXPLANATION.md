# DeepSeek V2 Mixture of Experts (MoE) - Detailed Explanation

## Overview

This document explains how the Mixture of Experts (MoE) system works in the DeepSeek V2 model implementation. The MoE architecture allows the model to dynamically select which expert networks to use for each token, making the model more efficient and scalable.

## Architecture Flow

### High-Level Flow Diagram

```
Input (hidden_states: [num_tokens, hidden_size])
    ↓
┌─────────────────────────────────────────┐
│  MoEGate (Router)                       │
│  - Linear layer: hidden → expert scores │
│  - Output: router_logits [num_tokens,   │
│    n_routed_experts]                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  TopK Selection                         │
│  - Softmax on router_logits             │
│  - Grouped top-k selection              │
│  - Output: topk_ids, topk_weights       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Expert Execution (FusedMoE)             │
│  - Dispatch tokens to experts           │
│  - Each expert processes its tokens     │
│  - Weighted combination of outputs      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Shared Experts (Always Active)         │
│  - Processes ALL tokens                  │
│  - Captures common patterns              │
└─────────────────────────────────────────┘
    ↓
final_hidden_states + shared_output → Output
```

### Visual Token Routing Example

```
Token 0: [0.1, 0.3, 0.05, ...] → TopK → [Expert 1, Expert 5, Expert 8]
Token 1: [0.2, 0.1, 0.4, ...] → TopK → [Expert 2, Expert 1, Expert 7]
Token 2: [0.05, 0.15, 0.3, ...] → TopK → [Expert 5, Expert 3, Expert 1]
...

After Dispatching:
Expert 0: [Token 5, Token 12, ...] → Process → Output 0
Expert 1: [Token 0, Token 1, Token 2, ...] → Process → Output 1
Expert 2: [Token 1, Token 8, ...] → Process → Output 2
...

Recombine with weights:
Final = Σ(weight_i × expert_i_output) + shared_expert_output
```

---

## 1. The MoEGate (Router) - How It Knows Which Expert to Call

### Location: `deepseek_v2.py` lines 336-406

The `MoEGate` class is the **router** that decides which experts should process each token.

### How It Works:

```python
class MoEGate(nn.Module):
    def __init__(self, config, quant_config, prefix, is_nextn):
        # Creates a weight matrix: (n_routed_experts, hidden_size)
        # This is a learnable parameter that maps hidden states to expert scores
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
```

**Key Components:**

1. **Weight Matrix (`self.weight`)**:
   - Shape: `(n_routed_experts, hidden_size)`
   - This is a **learnable linear layer** that projects each token's hidden state to a score for each expert
   - Each row corresponds to one expert
   - During training, this learns which features in the hidden state indicate which expert should be used

2. **Forward Pass**:
   ```python
   def forward(self, hidden_states, ...):
       # hidden_states shape: (num_tokens, hidden_size)
       # Computes: hidden_states @ weight.T
       logits = F.linear(hidden_states, self.weight, None)
       # Returns: (num_tokens, n_routed_experts)
       return logits
   ```

**What Happens:**
- Takes input `hidden_states` of shape `(num_tokens, hidden_size)`
- Multiplies with the gate weight matrix
- Produces `router_logits` of shape `(num_tokens, n_routed_experts)`
- Each value in `router_logits[i, j]` represents how "good" expert `j` is for token `i`

**Correction Bias (DeepSeek V2 specific):**
- If `config.topk_method == "noaux_tc"`, the gate includes `e_score_correction_bias`
- This is a learnable bias per expert: `(n_routed_experts,)`
- Used during top-k selection to adjust expert scores
- Helps balance expert usage during training

**Imports:**
- `F.linear` from `torch.nn.functional` (line 27)
- `nn.Parameter` from `torch.nn` (line 29)
- Custom kernel `dsv3_router_gemm` from `sgl_kernel` (line 199) - optimized CUDA kernel for router computation on certain hardware

---

## 2. TopK Selection - Choosing the Best Experts

### Location: `python/sglang/srt/layers/moe/topk.py`

The `TopK` class takes the router logits and selects the top-k experts for each token.

### How It Works:

```python
class TopK(MultiPlatformOp):
    def __init__(self, top_k, use_grouped_topk, ...):
        self.topk_config = TopKConfig(
            top_k=top_k,  # Number of experts to select per token
            use_grouped_topk=use_grouped_topk,  # DeepSeek V2 uses grouped topk
            ...
        )
```

**Key Function: `select_experts()`** (lines 851-988 in `topk.py`)

For DeepSeek V2, it uses **grouped top-k** selection:

1. **Grouped Top-K Logic** (`grouped_topk_gpu` function, lines 485-549):
   ```python
   # Step 1: Convert logits to probabilities
   scores = torch.softmax(router_logits, dim=-1)  # (num_tokens, num_experts)
   
   # Step 2: Group experts and find best groups
   # Experts are divided into groups (e.g., 64 experts in 8 groups = 8 experts per group)
   group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
   # Shape: (num_tokens, num_expert_groups)
   
   # Step 3: Select top-k groups
   group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
   
   # Step 4: Mask out experts not in selected groups
   # Step 5: Select top-k experts from the selected groups
   topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)
   ```

2. **Output Format**:
   - `topk_weights`: `(num_tokens, top_k)` - weights/probabilities for selected experts
   - `topk_ids`: `(num_tokens, top_k)` - indices of selected experts
   - `router_logits`: Original logits for reference

**Why Grouped Top-K?**
- DeepSeek V2 uses expert groups to ensure diversity
- Instead of just picking top-k experts globally, it:
  1. First selects the best groups
  2. Then picks experts within those groups
- This prevents all tokens from routing to the same few experts
- Ensures load balancing across experts

**Correction Bias in TopK:**
- If `correction_bias` is provided (from `MoEGate.e_score_correction_bias`):
  - Applied during expert selection: `scores_for_choice = scores + correction_bias`
  - Used for selection but not for final weights
  - Helps with expert load balancing

**Imports:**
- `torch` (line 31)
- `torch.topk` for selecting top-k values
- Custom kernels: `topk_softmax` from `sgl_kernel` (line 84)

---

## 3. Expert Execution - Running Selected Experts

### Location: `deepseek_v2.py` lines 462-477

The `experts` object is created using `get_moe_impl_class()`:

```python
self.experts = get_moe_impl_class(quant_config)(
    num_experts=config.n_routed_experts + self.num_fused_shared_experts,
    top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
    hidden_size=config.hidden_size,
    intermediate_size=config.moe_intermediate_size,
    ...
)
```

**What Happens:**

1. **Token Dispatching** (handled by `BaseDispatcher`):
   - Tokens are grouped by which expert they selected
   - All tokens going to expert 0 are batched together
   - All tokens going to expert 1 are batched together
   - This creates efficient batches for parallel processing
   - Example: If tokens [0, 5, 12] all selected expert 2, they're batched together

2. **Expert Processing**:
   - Each expert is a neural network (MLP) that processes its assigned tokens
   - Each expert has the same architecture:
     - `gate_up_proj`: Projects input from `hidden_size` to `intermediate_size`
     - Activation function (SiLU): `silu(gate) * up`
     - `down_proj`: Projects back from `intermediate_size` to `hidden_size`
   - All experts run in parallel (fused operation for efficiency)

3. **Recombination** (handled by `BaseDispatcher`):
   - Processed tokens are scattered back to their original positions
   - Each token's output is a weighted sum: `Σ(weight_i × expert_i_output)`
   - The weights come from `topk_weights` computed during selection

**Forward Pass** (lines 678-679):
```python
router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
topk_output = self.topk(hidden_states, router_logits)
final_hidden_states = self.experts(hidden_states, topk_output)
```

**Imports:**
- `get_moe_impl_class` from `sglang.srt.layers.moe.ep_moe.layer` (line 97)
- `FusedMoE` from `sglang.srt.layers.moe.fused_moe_triton.layer` (line 98)

---

## 4. Shared Experts - Always Active

### Location: `deepseek_v2.py` lines 505-555

DeepSeek V2 also has **shared experts** that process ALL tokens (not just selected ones):

```python
if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
    self.shared_experts = DeepseekV2MLP(...)
```

**Purpose:**
- Shared experts capture common patterns that apply to all tokens
- They're always active, unlike routed experts which are selectively activated
- Output is added to the routed expert output

**Forward Pass** (lines 674-676):
```python
shared_output = self._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
# Later combined with routed expert output
final_hidden_states += shared_output
```

---

## 5. Complete Forward Flow

### Location: `deepseek_v2.py` lines 658-722

Here's the complete flow in `forward_normal()`:

```python
def forward_normal(self, hidden_states, ...):
    # Step 1: Process through shared experts (if not fused)
    if not self._fuse_shared_experts_inside_sbo:
        shared_output = self._forward_shared_experts(hidden_states, ...)
    
    # Step 2: Compute router logits (scores for each expert)
    router_logits = self.gate(hidden_states, ...)
    # Shape: (num_tokens, n_routed_experts)
    
    # Step 3: Select top-k experts for each token
    topk_output = self.topk(hidden_states, router_logits)
    # Returns: topk_weights, topk_ids, router_logits
    
    # Step 4: Process tokens through selected experts
    final_hidden_states = self.experts(hidden_states, topk_output)
    # Shape: (num_tokens, hidden_size)
    
    # Step 5: Combine with shared expert output
    if shared_output is not None:
        final_hidden_states += shared_output
    
    # Step 6: Apply scaling factor for routed experts
    if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
        final_hidden_states *= self.routed_scaling_factor
    
    # Step 7: All-reduce across tensor parallel ranks (if needed)
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    
    return final_hidden_states
```

---

## 6. Key Imports and Their Sources

### From PyTorch:
- `torch`, `torch.nn` (line 26, 29)
- `torch.nn.functional as F` (line 27)

### From SGLang Layers:
- `TopK` from `sglang.srt.layers.moe.topk` (line 105)
- `get_moe_impl_class` from `sglang.srt.layers.moe.ep_moe.layer` (line 97)
- `FusedMoE` from `sglang.srt.layers.moe.fused_moe_triton.layer` (line 98)
- `BaseDispatcher`, `CombineInput`, `DispatchOutput` from `sglang.srt.layers.moe.token_dispatcher.base` (lines 100-104)

### From SGLang Distributed:
- `get_moe_expert_parallel_world_size` from `sglang.srt.distributed` (line 46)
- `tensor_model_parallel_all_reduce` from `sglang.srt.distributed` (line 50)

### From Custom Kernels:
- `dsv3_router_gemm` from `sgl_kernel` (line 199) - optimized CUDA kernel for router computation
- `topk_softmax` from `sgl_kernel` (line 84) - optimized top-k selection

---

## 7. How the Model "Learns" Which Expert for Which Task

The model doesn't explicitly know "expert 5 is for math" or "expert 3 is for code". Instead:

1. **Training Process**:
   - The `MoEGate.weight` matrix is randomly initialized
   - During training, backpropagation updates both:
     - The gate weights (to route tokens better)
     - The expert weights (to specialize on routed tokens)
   - Experts naturally specialize because they only see certain types of tokens

2. **Emergent Specialization**:
   - Expert 0 might see more tokens with certain patterns → learns to handle those
   - Expert 1 might see different patterns → learns different skills
   - This happens automatically through gradient descent

3. **Grouped Top-K Ensures Diversity**:
   - By selecting from different groups, the model ensures:
     - Not all tokens go to the same expert
     - Different experts get different workloads
     - Specialization can emerge naturally

---

## 8. Example Walkthrough

Let's trace a single token through the system:

**Input:** Token with `hidden_state` of shape `(1, 7168)`

1. **Gate Forward:**
   ```python
   router_logits = gate(hidden_state)  # Shape: (1, 64)
   # Example output: [0.1, 0.05, 0.3, 0.02, ..., 0.15]
   # These are raw logits (not probabilities yet)
   ```

2. **TopK Selection:**
   ```python
   topk_output = topk(hidden_state, router_logits)
   # Converts to probabilities via softmax
   # Selects top-k experts (e.g., k=6)
   # Returns:
   #   topk_ids: [2, 15, 8, 31, 5, 12]  # Expert indices
   #   topk_weights: [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]  # Weights
   ```

3. **Expert Processing:**
   ```python
   # Token is sent to experts 2, 15, 8, 31, 5, 12
   # Each expert processes the token independently
   # Outputs are weighted and combined:
   final = 0.25*expert_2(token) + 0.20*expert_15(token) + ...
   ```

4. **Shared Expert:**
   ```python
   shared = shared_experts(token)  # Always processes all tokens
   final += shared
   ```

5. **Output:**
   ```python
   return final  # Shape: (1, 7168)
   ```

---

## Summary

The MoE system works through:

1. **Gate (Router)**: Learns to score each expert for each token
2. **TopK**: Selects the best experts using grouped selection
3. **Experts**: Process tokens in parallel, each specializing on different patterns
4. **Shared Experts**: Always active, capture common patterns
5. **Combination**: Weighted combination of expert outputs + shared expert output

The "knowledge" of which expert for which task emerges during training through gradient descent, not through explicit programming.

