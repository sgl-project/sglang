# SPDX-License-Identifier: Apache-2.0

"""SRT-owned session control module for omni AR/multimodal_gen interleaving

    Layering:

    ARBackend: Autoregressive text/session backend used by omni orchestrator
      -> SRTBackedOmniSessionBridge: concrete translator from generic omni AR semantics into SRT session operations
        -> OmniSessionRuntime
             - model_policy: model-specific prompt/decode/accounting rules; runtime-aware hooks may ask the runtime to run SRT decode or mutate condition paths
             - OmniSRTSchedulerExecutor: concrete SRT scheduler execution owned by the runtime
                -> SRT scheduler
"""
