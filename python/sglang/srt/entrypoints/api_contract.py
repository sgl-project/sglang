"""The HTTP JSON contract gate: `proto/sglang/api/v1` is the schema for both
the Rust and the Python server, so a `/generate` body the Python dataclass
parse admitted but the schema rejects is refused here with the schema's own
error text. The generated decoder is the single source of that judgment,
except for a rule the schema cannot state: the only string
`return_hidden_states` takes is `"last"`.
"""

from typing import Any, Optional

from sglang.api.v1.api_types import GenerateRequest, JsonContractError
from sglang.srt.managers.req_types import get_return_hidden_states_mode


def generate_contract_error(body: Any) -> Optional[str]:
    """The contract violation in a parsed `/generate` JSON body, or None."""
    try:
        request = GenerateRequest.from_json_value(body)
    except JsonContractError as e:
        return str(e)
    # The schema takes any string for `return_hidden_states`. Refuse the
    # others here: the route's request parser reads one such as `"true"` as
    # a bool before the server code can check it.
    modes = request.return_hidden_states
    for mode in modes if isinstance(modes, list) else [modes]:
        if isinstance(mode, str):
            try:
                get_return_hidden_states_mode(mode)
            except ValueError as e:
                return str(e)
    return None
