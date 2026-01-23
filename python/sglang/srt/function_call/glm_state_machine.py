from sglang.srt.function_call.state_machine import XmlConfig
from sglang.srt.function_call.universal_state_machine import UniversalXmlStateMachine


class GlmStateMachine(UniversalXmlStateMachine):
    """State machine for parsing GLM XML-like tool call format."""

    def __init__(self):
        config = XmlConfig(
            tool_tag="tool_call",
            param_key_tag="arg_key",
            param_value_tag="arg_value",
        )
        super().__init__(config)
