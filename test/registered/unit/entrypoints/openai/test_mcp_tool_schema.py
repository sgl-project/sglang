import asyncio
import unittest
from enum import Enum
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import AsyncMock, patch

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)
from pydantic import BaseModel, Field

from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, trim_schema
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Mode(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class Address(BaseModel):
    city: str


class NullableEnumArgs(BaseModel):
    mode: Optional[Mode] = None


class NullableObjectArgs(BaseModel):
    address: Optional[Address] = None


class NullableArrayArgs(BaseModel):
    modes: Optional[list[Mode]] = None


class StructuredUnionArgs(BaseModel):
    value: Union[Mode, Address]


class RecursiveArgs(BaseModel):
    child: Optional["RecursiveArgs"] = None


class Filter(BaseModel):
    any_of: list["Filter"] = Field(default_factory=list)


class TestMCPToolSchema(CustomTestCase):
    def test_nullable_enum_keeps_allowed_values(self):
        schema = trim_schema(NullableEnumArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["mode"],
            {"type": "string", "enum": ["fast", "slow"]},
        )

    def test_nullable_object_keeps_properties_and_required_fields(self):
        schema = trim_schema(NullableObjectArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["address"],
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    def test_nullable_array_resolves_item_references(self):
        schema = trim_schema(NullableArrayArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["modes"],
            {
                "type": "array",
                "items": {"type": "string", "enum": ["fast", "slow"]},
            },
        )

    def test_structured_union_keeps_each_branch(self):
        schema = trim_schema(StructuredUnionArgs.model_json_schema())
        variants = schema["properties"]["value"]["oneOf"]
        self.assertEqual(variants[0], {"type": "string", "enum": ["fast", "slow"]})
        self.assertEqual(variants[1]["properties"], {"city": {"type": "string"}})
        self.assertEqual(variants[1]["required"], ["city"])

    def test_primitive_union_preserves_existing_harmony_shape(self):
        self.assertEqual(
            trim_schema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            {"type": ["string"]},
        )
        self.assertEqual(
            trim_schema({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
            {"type": ["integer", "string"]},
        )

    def test_local_reference_decodes_json_pointer_escapes(self):
        schema = trim_schema(
            {
                "$defs": {"a/b~c": {"type": "string", "enum": ["x"]}},
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/a~1b~0c"}},
            }
        )
        self.assertEqual(
            schema["properties"]["value"], {"type": "string", "enum": ["x"]}
        )

    def test_recursive_reference_stops_expanding(self):
        schema = trim_schema(RecursiveArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["child"], {"$ref": "#/$defs/RecursiveArgs"}
        )

    def test_list_recursive_tool_loads_and_renders(self):
        tool = SimpleNamespace(
            name="search",
            description="Find matches",
            inputSchema=Filter.model_json_schema(),
            annotations=None,
        )
        initialization = SimpleNamespace(
            serverInfo=SimpleNamespace(name="filters"), instructions="Search filters"
        )
        server = MCPToolServer()
        with patch(
            "sglang.srt.entrypoints.openai.tool_server.list_server_and_tools",
            new=AsyncMock(return_value=(initialization, SimpleNamespace(tools=[tool]))),
        ):
            asyncio.run(server.add_tool_server("localhost:8000"))

        self.assertTrue(server.has_tool("filters"))
        self.assertEqual(
            tool.inputSchema["properties"]["any_of"]["items"],
            {"$ref": "#/$defs/Filter"},
        )
        content = DeveloperContent.new().with_tools(
            server.get_tool_description("filters")
        )
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        conversation = Conversation.from_messages(
            [Message.from_role_and_content(Role.DEVELOPER, content)]
        )
        prompt = encoding.decode(
            encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        )
        self.assertIn("namespace filters", prompt)
        self.assertIn("type search", prompt)
        self.assertIn("any_of", prompt)

    def test_unresolved_reference_reports_an_error(self):
        with self.assertRaisesRegex(ValueError, "Unresolved.*reference"):
            trim_schema({"$ref": "#/$defs/missing"})

    def test_deep_reference_chain_reports_a_bounded_error(self):
        definitions = {
            str(index): {"$ref": f"#/$defs/{index + 1}"} for index in range(65)
        }
        definitions["65"] = {"type": "string"}
        with self.assertRaisesRegex(ValueError, "reference depth exceeds 64"):
            trim_schema({"$defs": definitions, "$ref": "#/$defs/0"})

    def test_external_reference_is_not_fetched(self):
        with self.assertRaisesRegex(ValueError, "local.*references"):
            trim_schema({"$ref": "https://example.com/schema.json"})


if __name__ == "__main__":
    unittest.main()
