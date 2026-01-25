from sglang.srt.function_call.state_machine import XmlConfig
from sglang.srt.function_call.universal_state_machine import UniversalXmlStateMachine


class Qwen3CoderStateMachine(UniversalXmlStateMachine):
    def __init__(self):
        config = XmlConfig(
            tool_tag="tool_call",
            tool_name_tag="function",
            param_tag="parameter",
            attr_sep="=",
        )
        super().__init__(config)
