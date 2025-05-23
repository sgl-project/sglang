class EBNFComposer:
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
        py_value ::= basic_number | basic_string | basic_array | "True" | "False" | "None"
        basic_any ::= basic_number | basic_string | basic_array | basic_object
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
    """

    @staticmethod
    def get_value_rule(prop: dict, is_pythonic: bool) -> str:
        if "enum" in prop:
            return " | ".join([f'"{v}"' for v in prop["enum"]])
        return "py_value" if is_pythonic else "json"

    @staticmethod
    def build_ebnf(
        tools,
        *,
        tool_calls_rule: str,
        call_rule_fmt: str,
        arguments_rule_fmt: str,
        key_value_fmt: str,
        is_pythonic: bool = False,
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: list of Tool
            tool_calls_rule: the top-level rule string (e.g. "tool_calls ::= ...")
            call_rule_fmt: format string for call_{name} rule (expects {name}, {arguments_rule})
            arguments_rule_fmt: format string for arguments_{name} rule (expects {arg_rules})
            key_value_fmt: format for key-value pairs (expects {key}, {valrule})
            is_pythonic: if True, use pythonic value rules
        """
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
                valrule = EBNFComposer.get_value_rule(prop, is_pythonic)
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

        if is_pythonic:
            lines.append(EBNFComposer.pythonic_grammar_ebnf_str)
        else:
            lines.append(EBNFComposer.json_grammar_ebnf_str)
        return "\n".join(lines)
