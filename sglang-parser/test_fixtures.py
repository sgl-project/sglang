from sglang.srt.entrypoints.openai.protocol import Tool, Function


def get_test_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        "days": {"type": "int"},
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="sql_interpreter",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "dry_run": {"type": "boolean"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="TodoWrite",
                parameters={
                    "type": "array",
                    "todos": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["content", "status"],
                    },
                },
            ),
        ),
    ]
