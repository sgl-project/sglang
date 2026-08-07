# SPDX-License-Identifier: Apache-2.0

"""session-scoped realtime state, control signals, and runtime-only helpers"""

import os

if os.environ.get("SGLANG_LIGHTWEIGHT_RUNTIME") != "1":
    from sglang.multimodal_gen.runtime.realtime.control_signals import (
        ControlScriptQueue,
        ControlSignal,
        ControlSignalQueue,
        ControlSignalSamplingParams,
        ControlStateQueue,
        ControlStateTransition,
        ParsedControlEventPayload,
        parse_control_event_payload,
    )
    from sglang.multimodal_gen.runtime.realtime.session import (
        BaseRealtimeState,
        RealtimeSession,
        RealtimeSessionCapacityError,
        RealtimeSessionCache,
    )
    from sglang.multimodal_gen.runtime.realtime.states import (
        RealtimeCameraControlState,
        RealtimeCausalDecodeState,
        RealtimeCausalDiTState,
        get_realtime_causal_dit_state,
    )

    __all__ = [
        "BaseRealtimeState",
        "ControlScriptQueue",
        "ControlSignal",
        "ControlSignalQueue",
        "ControlSignalSamplingParams",
        "ControlStateQueue",
        "ControlStateTransition",
        "ParsedControlEventPayload",
        "RealtimeCameraControlState",
        "RealtimeCausalDecodeState",
        "RealtimeCausalDiTState",
        "RealtimeSession",
        "RealtimeSessionCapacityError",
        "RealtimeSessionCache",
        "get_realtime_causal_dit_state",
        "parse_control_event_payload",
    ]
else:
    __all__: list[str] = []
