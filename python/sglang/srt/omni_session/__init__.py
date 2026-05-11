# SPDX-License-Identifier: Apache-2.0

"""SRT-owned session control module for omni AR/generation interleaving.

Layering:

    ARBackend
      -> OmniSessionBridge: bridge surface between generic omni orchestrator and SRT session runtime, translating ARBackend semantics into srt-session semantics
        -> OmniSessionRuntime
             - model_policy: model-specific prompt/decode/accounting rules
             - OmniSRTSchedulerExecutor: concrete SRT scheduler execution
                -> SRT scheduler
"""
