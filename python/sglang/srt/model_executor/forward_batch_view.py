"""Field views shared by forward-batch preparers.

Callers choose fields and own validation, batch construction, and derived state.
This module only slices the first axis; layout-specific positions, padding,
speculative inputs, and metadata planning stay with their respective preparers.
"""

from typing import Any


def slice_batch_field(value: Any, selection: slice) -> Any:
    """Slice an optional field without copying tensor storage.

    Callers read declared attributes directly and retain their own required-field
    and length checks. Tensors and host sequences keep ordinary slice semantics;
    None stays None. Positions with a different token axis stay caller-owned.
    """
    return None if value is None else value[selection]
