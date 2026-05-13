# SPDX-License-Identifier: Apache-2.0

"""SRT-owned session control module for omni AR/multimodal_gen interleaving.

This package owns the SRT session/KV state machine used by colocated omni
models. Do not confuse it with `sglang.omni.runtime`, which only wires omni
requests into the serve/scheduler runtime and does not execute ModelRunner KV
operations itself.

Layering:

ARBackend: Autoregressive text/session backend used by omni orchestrator
  -> SRTBackedOmniSessionAdapter: concrete translator from generic omni AR semantics into SRT session operations
    -> OmniSessionRuntime
         - model_policy: model-specific prompt/decode/accounting rules; runtime-aware hooks may ask the runtime to run SRT decode or mutate condition paths
         - OmniSRTSchedulerExecutor: concrete SRT scheduler execution owned by the runtime
            -> SRT scheduler
"""
