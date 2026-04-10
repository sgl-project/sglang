router should contain the original router logic and add one extra logic on choosing between precisions
refer to the reference for implementation guide! the reference's implementation is already tuned to be compatible for both 
    torch.compile
    cudagraph

router should add minimal logical overhead

---
## concrete sglang routing architecture (added during step 0.1 refinement)

### existing routing flow (DO NOT modify — insert heter logic AFTER this)
```
gate(hidden_states) → router_logits [num_tokens, num_experts]
    ↓
TopK.forward(hidden_states, router_logits) → TopKOutput(topk_weights, topk_ids, router_logits)
    file: python/sglang/srt/layers/moe/topk.py
    ↓
FusedMoE.forward(hidden_states, topk_output)
    ↓
dispatcher.dispatch(hidden_states, topk_output) → DispatchOutput
    file: python/sglang/srt/layers/moe/token_dispatcher/standard.py
```

### where heter precision selection happens
    AFTER TopK produces topk_ids (which experts are selected) and BEFORE dispatch
    the heter policy examines topk_ids to count tokens per expert across the batch
    then classifies experts into precision groups based on token count + config thresholds

### proposed insertion point: inside HeterFusedMoE.forward_impl()
```python
def forward_impl(self, hidden_states, topk_output):
    # 1. standard dispatch (unchanged)
    dispatch_output = self.dispatcher.dispatch(hidden_states, topk_output)
    
    # 2. NEW: classify experts into precision groups
    expert_assignment = self.policy.assign(
        topk_ids=topk_output.topk_ids,      # [num_tokens, top_k]
        topk_weights=topk_output.topk_weights,  # [num_tokens, top_k]
        num_experts=self.num_experts,
        group_ratios=self.group_ratios,       # e.g. [0.8, 0.2] for cold/hot
    )
    # expert_assignment.group_assignments[0] = list of cold expert IDs
    # expert_assignment.group_assignments[1] = list of hot expert IDs
    
    # 3. NEW: run separate group-GEMMs
    output = torch.zeros_like(hidden_states)
    for group_idx, expert_ids in enumerate(expert_assignment.group_assignments):
        group_output = self._run_group(dispatch_output, group_idx, expert_ids)
        output += group_output
    
    # 4. standard combine
    combine_input = CombineInput(hidden_states=output)
    return self.dispatcher.combine(combine_input)
```

### torch.compile / cudagraph compatibility
    the policy.assign() must be deterministic given the same topk_ids
    under cudagraph: the assignment is baked at capture time (same as TRT-LLM)
    the token-count-based policy uses only tensor ops (bincount, argsort) — compile-safe
    avoid python-level if/else branching on dynamic values inside the captured region