"""Capturers for runtime model state exposed via `meta_info`.

Each capturer taps into a producer (MoE topk, NSA indexer, ...) during forward,
buffers per-token / per-layer indices on host, and the scheduler returns them on
finish so the user can analyze model decisions offline.

To add a new capturer:
1. Add `state_capturer/<feature>.py` with a subclass of `BaseTopkCapturer` (if
   the data is topk-shaped) or a new base.
2. Wire a producer-side `if (cap := get_global_<feature>_capturer()) is not None:
   cap.capture(...)` hook at the layer's exit point.
3. Plumb `<feature>` field through `io_struct` / `schedule_batch` / scheduler /
   tokenizer / detokenizer / spec workers / disagg, mirroring `routed_experts`.
4. Add a `--enable-return-<feature>` flag in `server_args` and an `init_<feature>_
   capturer` in `model_runner`.
"""
