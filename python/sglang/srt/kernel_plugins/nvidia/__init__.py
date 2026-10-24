"""NVIDIA megakernels: whole-step dispatch for out-of-tree megakernel plugins.

Plugins register specs through the sglang.srt.plugins entry point. With
--enable-nvidia-megakernel the runtime validates the configuration, builds the
spec after the model is prepared, and routes covered steps through it; other
steps use the model forward.
"""
