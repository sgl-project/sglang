"""Paged experts: serve an MoE model whose routed experts do not fit in GPU memory.

Each MoE layer keeps K of its E routed experts in a K-slot GPU table (the layer's own expert
tensors, created with K rows) and holds all E experts in host memory. Routing is unchanged
(E-wide); every forward step:

* the distinct experts the step routes to are split into waves of at most K (a single wave
  whenever they fit, e.g. at decode);
* each wave pages its missing experts into free slots, remaps the logical expert ids to slots,
  masks every expert outside the wave, and runs the wrapped fused-MoE method without its final
  top-k sum;
* each routed (token, expert) output comes from the one wave that holds the expert, and the
  merged outputs go through the same top-k sum as the unpaged layer, so the result is
  bit-identical to it.

The pieces, each replaceable on its own:

* ``formats``   -- per quantization method: which tensors the checkpoint loads and which are
  paged, what happens after loading, and which runner contract applies;
* ``runners``   -- per runner backend: how the runner is configured to return per-expert
  outputs, and how the waves' outputs are merged and summed;
* ``store``     -- the host copy of every expert and the transfer into GPU slots;
* ``residency`` -- which expert sits in which slot, and how a step is split into waves;
* ``executor``  -- runs one step: page in, masked GEMM per wave, merge (host-planned waves, or
  one wave decided on the GPU);
* ``sizing``    -- K from the memory budget when ``--paged-experts-num-resident`` is unset;
* ``method``    -- the ``FusedMoEMethodBase`` that wires them into a ``FusedMoE`` layer.

With decode CUDA graphs on (full backend), a step whose routed entries fit the K slots (decode
up to K // top_k requests) decides and pages on the GPU, so the graph captures it. Every other
step plans its waves on the host; under a breakable prefill graph that runs as an eager break.
Scope: unquantized (bf16/fp16), block-quantized FP8 and GPTQ (Marlin) experts on a single GPU.
"""

from sglang.srt.layers.moe.paged_experts.method import (
    PagedExpertsMoEMethod,
    make_for_layer,
)

__all__ = ["PagedExpertsMoEMethod", "make_for_layer"]
