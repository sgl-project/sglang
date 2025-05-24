from typing import Literal, Optional


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

    JSON_TYPE_MAPPING = {
        "string": "basic_string",
        "number": "basic_number",
        "integer": "basic_number",
        "boolean": "basic_boolean",
        "null": "basic_null",
        "array": "basic_array",
        "object": "basic_object",
    }

    PYTHONIC_TYPE_MAPPING = {
        "string": "basic_string",
        "number": "basic_number",
        "integer": "basic_number",
        "boolean": '"True" | "False"',
        "null": '"None"',
        "array": "basic_array",
        "object": "basic_object",
    }

    @staticmethod
    def get_value_rule(
        prop: dict, function_format: Literal["pythonic", "json"] = "json"
    ) -> str:
        if "enum" in prop:
            return EBNFComposer._handle_enum(prop, function_format)

        if "type" in prop:
            return EBNFComposer._handle_type(prop, function_format)

        return function_format

    @staticmethod
    def _handle_enum(prop: dict, function_format: str) -> str:
        """Handle enum properties by formatting each value according to type and format."""
        enum_values = prop["enum"]
        prop_type = prop.get("type", "string")

        # Define formatters for different type/format combinations
        formatters = {
            ("string", "json"): lambda v: f'"\\"{v}\\""',
            ("string", "pythonic"): lambda v: f'"\\"{v}\\""',
            ("number", "json"): str,
            ("number", "pythonic"): str,
            ("integer", "json"): str,
            ("integer", "pythonic"): str,
            ("boolean", "json"): lambda v: "true" if v else "false",
            ("boolean", "pythonic"): lambda v: "True" if v else "False",
        }

        # Get the formatter or default to string handling
        formatter = formatters.get(
            (prop_type, function_format),
            formatters[("string", function_format)],  # Default to string handling
        )

        formatted_values = [formatter(value) for value in enum_values]
        enum_rule = " | ".join(formatted_values)

        # Wrap in parentheses if there are multiple values to ensure correct EBNF precedence
        if len(formatted_values) > 1:
            enum_rule = f"({enum_rule})"

        return enum_rule

    @staticmethod
    def _handle_type(prop: dict, function_format: str) -> str:
        """Handle type properties using the appropriate type mapping."""
        prop_type = prop["type"]
        type_mapping = (
            EBNFComposer.PYTHONIC_TYPE_MAPPING
            if function_format == "pythonic"
            else EBNFComposer.JSON_TYPE_MAPPING
        )

        if isinstance(prop_type, list):
            type_rules = [
                type_mapping[single_type]
                for single_type in prop_type
                if single_type in type_mapping
            ]
            return " | ".join(type_rules) if type_rules else function_format

        return type_mapping.get(prop_type, function_format)

    @staticmethod
    def build_ebnf(
        tools,
        *,
        call_rule_fmt: Optional[str] = None,
        function_format: Literal["pythonic", "json"] = "json",
        bot_token: Optional[str] = None,
        eot_token: Optional[str] = None,
        tool_call_separator: Optional[str] = None,
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
            tool_call_separator: The separator between multiple tool calls
        """
        # =================================================================
        # Step 1: Determine the root tool calls rule
        # =================================================================
        if bot_token and eot_token:
            if tool_call_separator:
                root_rule = f'"{bot_token}" function_call ( "{tool_call_separator}" function_call )* "{eot_token}"'
            else:
                root_rule = f'"{bot_token}" function_call "{eot_token}"'
        else:
            root_rule = EBNFComposer.TOOL_CALLS_MAP[function_format]

        # =================================================================
        # Step 2: Build the header rules
        # =================================================================
        ebnf_lines = [
            f"root ::= {root_rule}",
            "function_call ::= "
            + " | ".join([f"call_{tool.function.name}" for tool in tools]),
        ]

        # =================================================================
        # Step 3: Set up formatting templates
        # =================================================================
        call_template = (
            f"call_{{name}} ::= {call_rule_fmt}"
            if call_rule_fmt
            else EBNFComposer.CALL_RULE_MAP[function_format]
        )
        args_template = EBNFComposer.ARGUMENTS_RULE_MAP[function_format]
        key_value_template = EBNFComposer.KEY_VALUE_RULE_MAP[function_format]

        # =================================================================
        # Step 4: Build rules for each tool
        # =================================================================
        for tool in tools:
            tool_name = tool.function.name
            params = tool.function.parameters or {}
            properties = params.get("properties", {})
            required_props = set(params.get("required", []))

            # Build argument rules for this tool
            arg_rules = []
            for prop_name, prop_schema in properties.items():
                value_rule = EBNFComposer.get_value_rule(prop_schema, function_format)
                # Create key=value pair
                pair = key_value_template.format(key=prop_name, valrule=value_rule)

                if prop_name not in required_props:
                    pair = f"[ {pair} ]"

                arg_rules.append(pair)

            # Combine all argument rules
            combined_args = ' "," '.join(arg_rules) if arg_rules else ""
            arguments_rule = args_template.format(arg_rules=combined_args)

            # Add the function call rule and its arguments rule
            ebnf_lines.append(
                call_template.format(
                    name=tool_name, arguments_rule=f"arguments_{tool_name}"
                )
            )
            ebnf_lines.append(f"arguments_{tool_name} ::= {arguments_rule}")

        # =================================================================
        # Step 5: Add base grammar rules
        # =================================================================
        base_grammar = (
            EBNFComposer.pythonic_grammar_ebnf_str
            if function_format == "pythonic"
            else EBNFComposer.json_grammar_ebnf_str
        )
        ebnf_lines.append(base_grammar)

        return "\n".join(ebnf_lines)
