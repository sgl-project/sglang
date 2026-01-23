from sglang.srt.function_call.state_machine import JsonConfig
from sglang.srt.function_call.universal_state_machine import UniversalJsonStateMachine


class MistralStateMachine(UniversalJsonStateMachine):
    def __init__(self, is_compact: bool = False):
        if is_compact:
            config = JsonConfig(
                name_prefix="[TOOL_CALLS]", name_suffix="[ARGS]", is_array=False
            )
        else:
            config = JsonConfig(prefix="[TOOL_CALLS] [", suffix="]", is_array=True)
        super().__init__(config)
