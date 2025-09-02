# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""JSON Schema composer for tool calling constraints."""

from typing import Any, Dict, List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool


class JSONSchemaComposer:
    """Composer for generating JSON schemas for tool calling constraints.

    This class generates JSON schemas that can be used with XGrammar to constrain
    model outputs to valid tool calling formats. The approach is inspired by VLLM's
    implementation and provides a unified schema that works across different model formats.
    """

    @staticmethod
    def build_json_schema(
        tools: List[Tool],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
    ) -> Optional[Dict[str, Any]]:
        """Build a JSON schema for tool calling constraints.

        Args:
            tools: List of Tool objects to generate schema for
            tool_choice: Tool choice specification. Can be:
                - "none": No tools allowed
                - "auto": Tools are optional (model can choose to use or not)
                - "required": At least one tool call required
                - Dict with function name: Specific named tool required

        Returns:
            JSON schema dict or None if no tools allowed
        """
        # Handle no tools case
        if tool_choice == "none" or tools is None or len(tools) == 0:
            return None

        # Handle named tool case
        if isinstance(tool_choice, dict) and "function" in tool_choice:
            tool_name = tool_choice["function"]["name"]
            # Find the specific tool
            target_tool = None
            for tool in tools:
                if tool.function.name == tool_name:
                    target_tool = tool
                    break

            if target_tool is None:
                raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`.")

            return JSONSchemaComposer._build_named_tool_schema(target_tool)

        # Handle auto mode (tools are optional)
        if tool_choice == "auto":
            return JSONSchemaComposer._build_auto_schema(tools)

        # Handle required tools case (default)
        if tool_choice == "required":
            return JSONSchemaComposer._build_required_schema(tools)

        # Default to required if tool_choice is not recognized
        return JSONSchemaComposer._build_required_schema(tools)

    @staticmethod
    def _build_tool_schema(tool: Tool) -> Dict[str, Any]:
        """Generate schema for a single tool."""
        return {
            "properties": {
                "name": {"type": "string", "const": tool.function.name},
                "parameters": tool.function.parameters
                or {"type": "object", "properties": {}},
            },
            "required": ["name", "parameters"],
        }

    @staticmethod
    def _extract_defs_from_tools(tools: List[Tool]) -> Dict[str, Any]:
        """Extract and merge $defs from all tool parameters."""
        all_defs = {}
        for tool in tools:
            if tool.function.parameters and "$defs" in tool.function.parameters:
                tool_defs = tool.function.parameters["$defs"]
                for def_name, def_schema in tool_defs.items():
                    if def_name not in all_defs:
                        all_defs[def_name] = def_schema
                    elif all_defs[def_name] != def_schema:
                        raise ValueError(
                            f"Tool definition '{def_name}' has multiple schemas, "
                            "which is not supported."
                        )
        return all_defs

    @staticmethod
    def _build_array_schema(tools: List[Tool], min_items: int) -> Dict[str, Any]:
        """Build array schema with specified minimum items."""
        json_schema = {
            "type": "array",
            "minItems": min_items,
            "items": {
                "type": "object",
                "anyOf": [
                    JSONSchemaComposer._build_tool_schema(tool) for tool in tools
                ],
            },
        }

        # Add $defs if present
        all_defs = JSONSchemaComposer._extract_defs_from_tools(tools)
        if all_defs:
            json_schema["$defs"] = all_defs

        return json_schema

    @staticmethod
    def _build_named_tool_schema(tool: Tool) -> Dict[str, Any]:
        """Build schema for a specific named tool.

        Args:
            tool: The specific tool to generate schema for

        Returns:
            JSON schema for the named tool
        """
        return {"type": "object", **JSONSchemaComposer._build_tool_schema(tool)}

    @staticmethod
    def _build_required_schema(tools: List[Tool]) -> Dict[str, Any]:
        """Build schema for required tool calling (at least one tool).

        Args:
            tools: List of tools to generate schema for

        Returns:
            JSON schema for required tool calling
        """
        return JSONSchemaComposer._build_array_schema(tools, min_items=1)

    @staticmethod
    def _build_auto_schema(tools: List[Tool]) -> Dict[str, Any]:
        """Build schema for auto tool calling (tools are optional).

        Args:
            tools: List of tools to generate schema for

        Returns:
            JSON schema for auto tool calling (optional tool usage)
        """
        return JSONSchemaComposer._build_array_schema(tools, min_items=0)

    @staticmethod
    def _validate_tool_parameters(parameters: Dict[str, Any]) -> None:
        """Validate that tool parameters conform to expected JSON Schema format.

        Args:
            parameters: Tool parameters dict to validate

        Raises:
            ValueError: If parameters are not valid JSON Schema
        """
        if not isinstance(parameters, dict):
            raise ValueError("Tool parameters must be a dictionary")

        # Check for required JSON Schema fields
        if "type" not in parameters:
            raise ValueError("Tool parameters must have a 'type' field")

        # Validate type field
        valid_types = [
            "object",
            "array",
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        ]
        if parameters["type"] not in valid_types:
            raise ValueError(
                f"Invalid type '{parameters['type']}'. Must be one of: {valid_types}"
            )

        # If type is object, validate properties structure
        if parameters["type"] == "object":
            if "properties" in parameters:
                if not isinstance(parameters["properties"], dict):
                    raise ValueError("Properties must be a dictionary")

                # Validate each property
                for prop_name, prop_schema in parameters["properties"].items():
                    if not isinstance(prop_schema, dict):
                        raise ValueError(
                            f"Property '{prop_name}' schema must be a dictionary"
                        )
                    JSONSchemaComposer._validate_tool_parameters(prop_schema)

            # Validate required fields
            if "required" in parameters:
                if not isinstance(parameters["required"], list):
                    raise ValueError("Required must be a list")
                if not all(isinstance(req, str) for req in parameters["required"]):
                    raise ValueError("All required items must be strings")

        # If type is array, validate items
        elif parameters["type"] == "array":
            if "items" in parameters:
                JSONSchemaComposer._validate_tool_parameters(parameters["items"])

        # Validate enum if present
        if "enum" in parameters:
            if not isinstance(parameters["enum"], list):
                raise ValueError("Enum must be a list")
            if len(parameters["enum"]) == 0:
                raise ValueError("Enum cannot be empty")

        # Validate format if present
        if "format" in parameters:
            valid_formats = ["date", "date-time", "email", "uri", "uuid"]
            if parameters["format"] not in valid_formats:
                raise ValueError(
                    f"Invalid format '{parameters['format']}'. Must be one of: {valid_formats}"
                )

        # Validate pattern if present
        if "pattern" in parameters:
            if not isinstance(parameters["pattern"], str):
                raise ValueError("Pattern must be a string")
            # Could add regex validation here if needed

    @staticmethod
    def build_schema_with_validation(
        tools: List[Tool],
        tool_choice: Union[str, Dict[str, Any]] = "required",
        validate_parameters: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Build JSON schema with optional parameter validation.

        Args:
            tools: List of Tool objects to generate schema for
            tool_choice: Tool choice specification
            validate_parameters: Whether to validate tool parameters

        Returns:
            JSON schema dict or None if no tools allowed

        Raises:
            ValueError: If validation fails and validate_parameters is True
        """
        if validate_parameters:
            for tool in tools:
                if tool.function.parameters:
                    JSONSchemaComposer._validate_tool_parameters(
                        tool.function.parameters
                    )

        return JSONSchemaComposer.build_json_schema(tools, tool_choice)
