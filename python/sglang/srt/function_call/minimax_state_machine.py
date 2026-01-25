from sglang.srt.function_call.state_machine import XmlConfig
from sglang.srt.function_call.universal_state_machine import UniversalXmlStateMachine


class MinimaxStateMachine(UniversalXmlStateMachine):
    """State machine for parsing Minimax XML-like tool call format."""

    def __init__(self):
        config = XmlConfig(
            root_tag="minimax:tool_call",
            tool_tag="invoke",
            tool_name_attr="name",
            param_tag="parameter",
            param_name_attr="name",
        )
        super().__init__(config)
