import unittest

from sglang.srt.function_call.state_machine import (
    JsonConfig,
    UniversalToolParserState,
    XmlConfig,
)
from sglang.srt.function_call.universal_state_machine import (
    UniversalJsonStateMachine,
    UniversalXmlStateMachine,
)


class TestUniversalXmlStateMachine(unittest.TestCase):
    def test_minimax_style(self):
        config = XmlConfig(
            root_tag="minimax:tool_call",
            tool_tag="invoke",
            tool_name_attr="name",
            param_tag="parameter",
            param_name_attr="name",
        )
        sm = UniversalXmlStateMachine(config)

        data = '<minimax:tool_call><invoke name="get_weather"><parameter name="city">San Francisco</parameter></invoke></minimax:tool_call>'
        result = sm.parse(data)
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(len(result.completed_tools), 1)
        self.assertEqual(result.completed_tools[0]["name"], "get_weather")
        self.assertEqual(
            result.completed_tools[0]["arguments"], {"city": "San Francisco"}
        )

    def test_glm_style(self):
        config = XmlConfig(
            tool_tag="tool_call", param_key_tag="arg_key", param_value_tag="arg_value"
        )
        sm = UniversalXmlStateMachine(config)

        data = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        result = sm.parse(data)
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(result.completed_tools[0]["name"], "get_weather")
        self.assertEqual(result.completed_tools[0]["arguments"], {"city": "Beijing"})

    def test_qwen_style(self):
        config = XmlConfig(
            tool_tag="tool_call",
            tool_name_tag="function",
            attr_sep="=",
            param_tag="parameter",
        )
        sm = UniversalXmlStateMachine(config)

        data = "<tool_call><function=get_weather><parameter=city>Shanghai</parameter></tool_call>"
        result = sm.parse(data)
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(result.completed_tools[0]["name"], "get_weather")
        self.assertEqual(result.completed_tools[0]["arguments"], {"city": "Shanghai"})


class TestUniversalJsonStateMachine(unittest.TestCase):
    def test_mistral_style(self):
        config = JsonConfig(prefix="[TOOL_CALLS]", is_array=True)
        sm = UniversalJsonStateMachine(config)

        data = '[TOOL_CALLS] {"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = sm.parse(data)
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(len(result.completed_tools), 1)
        self.assertEqual(result.completed_tools[0]["name"], "get_weather")

    def test_markdown_json(self):
        config = JsonConfig(prefix="```json", suffix="```", is_array=False)
        sm = UniversalJsonStateMachine(config)

        data = 'Thought: I should call weather. ```json\n{"name": "get_weather", "arguments": {"city": "London"}}\n```'
        result = sm.parse(data)
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(result.completed_tools[0]["name"], "get_weather")
        self.assertEqual(result.normal_text, "Thought: I should call weather. ")


if __name__ == "__main__":
    unittest.main()
