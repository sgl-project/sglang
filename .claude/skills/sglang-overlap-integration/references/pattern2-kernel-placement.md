# Pattern 2: Self-Contained Kernel Placement (Detail)

**Rule: overlap kernel code must be self-contained — no external third-party dependency.**

## Placement

Kernel files go under:
```
python/sglang/srt/distributed/device_communicators/symm_mem_kernels/<kernel_name>_symm_mem.py
```

Add public API exports to `symm_mem_kernels/__init__.py`.

## Context object pattern

Each kernel exposes three items:

| Item | Role |
|------|------|
| Context class | Holds symm-mem buffers, signals, config; **must implement `finalize()`** for cleanup |
| Context factory | Creates the context; called by communicator lazy-init |
| Op function | Executes the kernel given context + input tensors |

The context class holds all persistent state (symmetric memory buffers, rendezvous handles,
signal buffers). `finalize()` is called when shape parameters change and the context must
be rebuilt.

## Context class conventions

- Use `@dataclass` for the context class
- All symm-mem buffers and signal tensors are instance attributes
- `__post_init__` performs the actual symm-mem allocation and rendezvous
- Expose pointer arrays (`buf_ptrs`, `signal_pad_ptrs`) as `torch.Tensor` of dtype `int64`
  for GPU-side peer indexing (e.g., `tl.load(ptrs + peer)` in Triton kernels)
- `finalize()` releases all symm-mem resources and resets handles to `None`

## Op function conventions

- Takes `ctx` as first argument, followed by input tensors and runtime parameters
- Resets synchronization state before launch (e.g., `ctx.grid_barrier.zero_()`)
- Returns the output tensor(s) — same shape/dtype as the original sequential path would produce
- Does NOT modify the context's persistent buffers in a way that would break reuse
