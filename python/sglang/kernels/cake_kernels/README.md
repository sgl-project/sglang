# Cake kernels

This directory owns thin adapters to Cake-generated implementations distributed
by FlashInfer. Generated CUDA sources and JIT caches remain owned by FlashInfer.
As with `kda_kernels/`, runtime consumers import the operator facade under
`sglang.kernels.ops`; importing the namespace must not initialize a backend.

Backend selection must preserve the operator's dtype, layout, state, mutation,
and numerical contract. In particular, Cake top-k-then-top-p sampling is not a
drop-in replacement for joint top-k/top-p filtering. Prepared plans must remain
valid when input contents and buffers change across calls and CUDA Graph replay.

In `sampling.softmax`, SM100-family GPUs run the JIT softmax first. The Cake
softmax runs only when the JIT softmax rejects the input and the logits are FP32
and contiguous on SM100 or SM103, with a small batch and a large vocabulary. The
sampler retains its existing filtering and random-number generation, and
deterministic inference retains torch.softmax.
Other operators require their own numerical and real-model performance evidence
before being added here.

The Cake softmax in `sampling.softmax` also requires FlashInfer's
`cake_blackwell_softmax` module. With an older FlashInfer, that path uses Torch
softmax. This adapter does not change SGLang's dependency pins.

DSpark's draft softmax (`SoftmaxTemp` in
`ops/speculative/dspark/dspark_accept.py`) calls FlashInfer's softmax directly
and does not use this adapter. FlashInfer routes that call to its Cake softmax
on SM100 and SM103.
