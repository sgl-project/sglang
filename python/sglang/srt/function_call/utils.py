import json
from json import JSONDecodeError, JSONDecoder
from typing import Any, Tuple, Union, Literal, Optional, List
from sglang.srt.entrypoints.openai.protocol import ToolChoice, Tool

import partial_json_parser
from partial_json_parser.core.options import Allow


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False

def _get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]]
) -> Optional[dict]:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification

    Returns:
        JSON schema dict, or None if no valid tools found
    """
    def get_tool_schema(tool):
        return {
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
                if tool.function.parameters else {
                    "type": "object",
                    "properties": {}
                }
            },
            "required": ["name", "parameters"]
        }
    
    if isinstance(tool_choice, ToolChoice) or (isinstance(tool_choice, dict) and "function" in tool_choice):
        # For specific function choice, return the user's parameters schema directly
        if isinstance(tool_choice, ToolChoice):
            fn_name = tool_choice.function.name
        else:
            fn_name = tool_choice["function"]["name"]
        for tool in tools:
            if tool.function.name == fn_name:
                return {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": get_tool_schema(tool)
                }
        return None
    elif tool_choice == "required":

        def get_tool_schema_defs(tools):
            all_defs = {}
            for tool in tools:
                if tool.function.parameters is None:
                    continue
                # Make a copy to avoid modifying original
                params = tool.function.parameters.copy()
                defs = params.pop("$defs", {})
                for def_name, def_schema in defs.items():
                    if def_name in all_defs and all_defs[def_name] != def_schema:
                        raise ValueError(
                            f"Tool definition '{def_name}' has "
                            "multiple schemas, which is not "
                            "supported.")
                    else:
                        all_defs[def_name] = def_schema
            return all_defs

        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [get_tool_schema(tool) for tool in tools]
            }
        }
        json_schema_defs = get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None
