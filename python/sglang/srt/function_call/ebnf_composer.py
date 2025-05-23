from typing import Literal


class EBNFComposer:
    # Adapted from https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html#try-out-via-hf-transformers
    json_grammar_ebnf_str = r"""
        json ::= basic_array | basic_object
        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
        basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_boolean ::= "true" | "false"
        basic_null ::= "null"
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
        """

    pythonic_grammar_ebnf_str = r"""
        pythonic ::= basic_number | basic_string | basic_array | "True" | "False" | "None"
        basic_any ::= basic_number | basic_string | basic_array | basic_object
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
    """

    TOOL_CALLS_MAP = {
        "pythonic": '"[" function_call ("," function_call)* "]"',
        "json": "function_call",
    }

    CALL_RULE_MAP = {
        "pythonic": 'call_{name} ::= "{name}" "(" {arguments_rule} ")"',
        "json": 'call_{name} ::= "{{" "\\"name\\"" ":" "\\"{name}\\"" ", " "\\"arguments\\"" ":" {arguments_rule} "}}"',
    }

    ARGUMENTS_RULE_MAP = {
        "pythonic": "{arg_rules}",
        "json": '"{{" {arg_rules} "}}"',
    }

    KEY_VALUE_RULE_MAP = {
        "pythonic": '"{key}" "=" {valrule}',
        "json": '"\\"{key}\\"" ":" {valrule}',
    }

    @staticmethod
    def get_value_rule(
        prop: dict, function_format: Literal["pythonic", "json"] = "json"
    ) -> str:
        if "enum" in prop:
            return " | ".join([f'"{v}"' for v in prop["enum"]])
        return function_format

    @staticmethod
    def build_ebnf(
        tools,
        *,
        call_rule_fmt: str = None,
        function_format: Literal["pythonic", "json"] = "json",
        bot_token: str = None,
        eot_token: str = None,
        tool_call_separator: str = None,
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: List of Tool objects to generate EBNF grammar for
            call_rule_fmt: Optional custom format string for call_{name} rule. It should define each function call's format, with
                the placeholders {name} for the function name and {arguments_rule} for the arguments rule. If None, a default
                format based on function_format will be used.
            function_format: The format of function calls, either "pythonic" or "json"
            bot_token: The token that indicates the start of a tool call section
            eot_token: The token that indicates the end of a tool call section
            tool_call_separator: The separator between multiple tool calls (default: ",")
        """
        tool_calls_rule = None
        if bot_token is not None:
            if eot_token is not None:
                if tool_call_separator is not None:
                    tool_calls_rule = f'"{bot_token}" function_call ( "{tool_call_separator}" function_call )* "{eot_token}"'
                else:
                    tool_calls_rule = f'"{bot_token}" function_call "{eot_token}"'

        tool_calls_rule = (
            EBNFComposer.TOOL_CALLS_MAP[function_format]
            if not tool_calls_rule
            else tool_calls_rule
        )
        call_rule_fmt = (
            ("call_{name} ::= " + call_rule_fmt)
            if call_rule_fmt
            else EBNFComposer.CALL_RULE_MAP[function_format]
        )
        arguments_rule_fmt = EBNFComposer.ARGUMENTS_RULE_MAP[function_format]
        key_value_fmt = EBNFComposer.KEY_VALUE_RULE_MAP[function_format]

        lines = [
            "root ::= " + tool_calls_rule,
            "function_call ::= "
            + " | ".join([f"call_{t.function.name}" for t in tools]),
        ]
        for tool in tools:
            name = tool.function.name
            params = tool.function.parameters or {}
            props = list(params.get("properties", {}).keys())
            required = set(params.get("required", []))
            arg_rules = []
            for key in props:
                prop = params["properties"][key]
                valrule = EBNFComposer.get_value_rule(prop, function_format)
                pair = key_value_fmt.format(key=key, valrule=valrule)
                if key not in required:
                    pair = f"[ {pair} ]"
                arg_rules.append(pair)
            arguments_rule = arguments_rule_fmt.format(
                arg_rules=(' "," '.join(arg_rules) if arg_rules else "")
            )
            lines.append(
                call_rule_fmt.format(name=name, arguments_rule=f"arguments_{name}")
            )
            lines.append(f"arguments_{name} ::= {arguments_rule}")

        if function_format == "pythonic":
            lines.append(EBNFComposer.pythonic_grammar_ebnf_str)
        else:
            lines.append(EBNFComposer.json_grammar_ebnf_str)
        return "\n".join(lines)
