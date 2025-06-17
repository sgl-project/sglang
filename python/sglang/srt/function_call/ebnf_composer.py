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
        function_format: Literal["pythonic", "json"] = "json",
        # Parameters for wrapping the entire sequence of tool calls
        sequence_start_token: Optional[str] = None,
        sequence_end_token: Optional[str] = None,
        # Parameters for wrapping individual tool calls
        individual_call_start_token: Optional[str] = None,
        individual_call_end_token: Optional[str] = None,
        # Parameter for separating multiple tool calls
        tool_call_separator: Optional[str] = None,
        call_rule_fmt: Optional[str] = None,
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: List of Tool objects to generate EBNF grammar for
            function_format: The format of function calls, either "pythonic" or "json"
            sequence_start_token: Token that wraps the entire sequence of tool calls (start)
            sequence_end_token: Token that wraps the entire sequence of tool calls (end)
            individual_call_start_token: Token that wraps each individual tool call (start)
            individual_call_end_token: Token that wraps each individual tool call (end)
            tool_call_separator: The separator between multiple tool calls
            call_rule_fmt: Optional custom format string for call_{name} rule. It should define each function call's format, with
                the placeholders {name} for the function name and {arguments_rule} for the arguments rule. If None, a default
                format based on function_format will be used.
        """
        # =================================================================
        # Step 1: Determine the root tool calls rule
        # =================================================================
        # Handle a single function call
        if individual_call_start_token and individual_call_end_token:
            function_call_unit = f'"{individual_call_start_token}" function_call "{individual_call_end_token}"'
        else:
            function_call_unit = "function_call"

        # Handle multiple function calls with separators
        if tool_call_separator is not None:
            base_pattern = f'{function_call_unit} ( "{tool_call_separator}" {function_call_unit} )*'
        else:
            # Assume only support single function call
            base_pattern = function_call_unit

        # Apply sequence-level wrapping if needed
        if sequence_start_token and sequence_end_token:
            root_rule = (
                f'"{sequence_start_token}" {base_pattern} "{sequence_end_token}"'
            )
        else:
            root_rule = base_pattern

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

            # The generated pattern ensures:
            # 1. Required properties appear first, joined by commas
            # 2. Optional properties are wrapped with comma included: ( "," ( "prop" : value )? )?
            # 3. For multiple optional properties, we allow flexible ordering:
            #    - Each optional can be skipped entirely
            #    - They can appear in any combination
            #
            # Example patterns generated:
            # - One required, one optional:
            #   "{" "location" ":" string ( "," ( "unit" ":" enum ) )? "}"
            #   Allows: {"location": "Paris"} or {"location": "Paris", "unit": "celsius"}
            #
            # - Multiple optional properties with flexible ordering:
            #   "{" "req" ":" string ( "," ( "opt1" ":" value ( "," "opt2" ":" value )? | "opt2" ":" value ) )? "}"
            #   Allows: {"req": "x"}, {"req": "x", "opt1": "y"}, {"req": "x", "opt2": "z"},
            #           {"req": "x", "opt1": "y", "opt2": "z"}
            #
            # - All optional properties with flexible ordering:
            #   "{" ( "opt1" ":" value ( "," "opt2" ":" value )? | "opt2" ":" value )? "}"
            #   Allows: {}, {"opt1": "x"}, {"opt2": "y"}, {"opt1": "x", "opt2": "y"}

            prop_kv_pairs = {}
            ordered_props = list(properties.keys())

            for prop_name, prop_schema in properties.items():
                value_rule = EBNFComposer.get_value_rule(prop_schema, function_format)
                # Create key=value pair
                pair = key_value_template.format(key=prop_name, valrule=value_rule)
                prop_kv_pairs[prop_name] = pair

            # Separate into required and optional while preserving order
            required = [p for p in ordered_props if p in required_props]
            optional = [p for p in ordered_props if p not in required_props]

            # Build the combined rule
            rule_parts = []

            # Add required properties joined by commas
            if required:
                rule_parts.append(' "," '.join(prop_kv_pairs[k] for k in required))

            # Add optional properties with flexible ordering
            if optional:
                # Build alternatives where any optional property can appear first
                opt_alternatives = []
                for i in range(len(optional)):
                    # Build pattern for optional[i] appearing first
                    opt_parts = []
                    for j in range(i, len(optional)):
                        if j == i:
                            opt_parts.append(prop_kv_pairs[optional[j]])
                        else:
                            opt_parts.append(f' ( "," {prop_kv_pairs[optional[j]]} )?')
                    opt_alternatives.append("".join(opt_parts))

                # Wrap with appropriate comma handling based on whether we have required properties
                if required:
                    # Required properties exist, so optional group needs outer comma
                    rule_parts.append(' ( "," ( ')
                    rule_parts.append(" | ".join(opt_alternatives))
                    rule_parts.append(" ) )?")
                else:
                    # All properties are optional
                    rule_parts.append("( ")
                    rule_parts.append(" | ".join(opt_alternatives))
                    rule_parts.append(" )?")

            combined_args = "".join(rule_parts)
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
