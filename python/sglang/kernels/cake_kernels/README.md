# Cake kernels

This directory owns thin adapters to Cake-generated implementations distributed
by FlashInfer. Generated CUDA sources and JIT caches remain owned by FlashInfer.
As with `kda_kernels/`, runtime consumers import the operator facade under
`sglang.kernels.ops`; importing the namespace must not initialize a backend.

Backend selection must preserve the operator's dtype, layout, state, mutation,
and numerical contract. In particular, Cake top-k-then-top-p sampling is not a
drop-in replacement for joint top-k/top-p filtering. Prepared plans must remain
valid when input contents and buffers change across calls and CUDA Graph replay.

The sampler's softmax fast path accepts FP32 contiguous logits on SM103,
small batches and large vocabularies. The sampler retains its existing filtering
and random-number generation, and deterministic inference retains torch.softmax.
Other operators require their own numerical and real-model performance evidence
before being added here.

The fast path also requires FlashInfer's `cake_blackwell_softmax` module. Older
FlashInfer installations retain Torch softmax; this adapter does not change
SGLang's dependency pins.
