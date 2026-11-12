"""The HTTP JSON contract gate: `proto/sglang/api/v1` is the schema for both
the Rust and the Python server, so a `/generate` body the Python dataclass
parse admitted but the schema rejects is refused here with the schema's own
error text. The generated decoder is the single source of that judgment.
"""

from typing import Any, Optional

from sglang.api.v1.api_types import GenerateRequest, JsonContractError

# The schema is shared with the Rust server, which is the only one that acts on
# this field. Refuse it here so a client is not handed decoded text it asked
# the server to skip.
SKIP_DETOKENIZE_UNSUPPORTED = (
    "skip_detokenize is only supported by the Rust server; "
    "this server would still decode the output text"
)


def generate_contract_error(body: Any) -> Optional[str]:
    """The contract violation in a parsed `/generate` JSON body, or None."""
    try:
        request = GenerateRequest.from_json_value(body)
    except JsonContractError as e:
        return str(e)
    if request.skip_detokenize:
        return SKIP_DETOKENIZE_UNSUPPORTED
    return None
