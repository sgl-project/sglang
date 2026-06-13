"""Public compatibility API for function-call parsing.

The implementation is split by responsibility:

- ``mode.py`` defines the compatibility policy, events, records, and strict
  mode behavior.
- ``recovery.py`` owns fail-open recovery at the ``FunctionCallParser``
  boundary.
- ``param_types.py`` handles schema-driven value conversion.

Importing from ``sglang.srt.function_call.compatibility`` remains the stable
package-level API for detectors and tests.
"""

from sglang.srt.function_call.compatibility.mode import (
    CompatibilityEvent,
    CompatibilityMode,
    CompatibilityRecord,
    CompatibilityViolation,
)
from sglang.srt.function_call.compatibility.recovery import (
    recover_nonstream,
    recover_stream,
    synthesize_json_close,
)

__all__ = [
    "CompatibilityEvent",
    "CompatibilityMode",
    "CompatibilityRecord",
    "CompatibilityViolation",
    "recover_nonstream",
    "recover_stream",
    "synthesize_json_close",
]
