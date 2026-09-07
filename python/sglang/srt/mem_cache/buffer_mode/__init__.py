"""Buffer-only HiCache host memory mode (--hicache-host-memory-mode buffer_only).

Host RAM is a transient staging buffer between the GPU and the L3 storage
backend, never an L2 cache tier: writes stage device KV through op-owned
host bounces into storage and free them at the storage ack; reads fetch
storage hits into op-owned bounces and publish them into the device tree at
prefill admission. ``UnifiedRadixCache`` composes ``BufferModePipeline`` for
the two transfer pipelines and dispatches to it at the mode branches.
"""
