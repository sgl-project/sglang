# TCPCG reference only

Snapshot from Oasis-Git/sglang main at c421d16563, before TCPCG removal.

These files preserve the old compiler adapters, FX partitioning, and per-piece
capture/replay implementation for future design reference. They are outside the installed SGLang
package, are not imported by the serving runtime, and are not a supported or
runnable backend. Imports intentionally retain their historical names; the
runtime interfaces and model split ops they referenced have been removed.

Active torch.compile support remains under python/sglang/srt/compilation.
Shared graph tensor-lifetime helpers live under runner_backend_utils.
