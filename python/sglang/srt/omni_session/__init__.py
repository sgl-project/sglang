# SPDX-License-Identifier: Apache-2.0

"""SRT-owned session control module for omni AR/multimodal_gen interleaving.

This package owns the SRT session/KV state machine used by colocated omni
models. Do not confuse it with `sglang.omni.runtime`, which only wires omni
requests into the serve/scheduler runtime and does not execute ModelRunner KV
operations itself.

Layering:

ARBackend: Autoregressive text/session backend used by omni coordinator
  -> SRTBackedOmniSessionAdapter: concrete translator from generic omni AR semantics into SRT session operations
    -> OmniSessionRuntime
         - model_hooks: model-specific prompt/decode/accounting hooks
         - OmniSRTSchedulerExecutor: concrete SRT scheduler execution owned by the runtime
            -> SRT scheduler

Session adapters own request-level choices such as mode, think flag, condition
paths, and generated-media commit policy. Model hooks own token grammar and the
small callbacks that turn those choices into SRT prepared inputs, segment
boundaries, VLM decode, and model-state patches.
"""
