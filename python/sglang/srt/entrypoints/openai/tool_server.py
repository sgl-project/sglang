# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.types import ListToolsResult
except ImportError as e:
    ClientSession = sse_client = ListToolsResult = e

from openai_harmony import ToolDescription, ToolNamespaceConfig

logger = logging.getLogger(__name__)


async def list_server_and_tools(server_url: str):

    async with (
        sse_client(url=server_url) as streams,
        ClientSession(*streams) as session,
    ):
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response


def trim_schema(schema: dict) -> dict:
    """Adapt MCP JSON Schema to Harmony's descriptive schema format."""
    root = schema

    def normalize(node, seen_refs=()):
        if not isinstance(node, dict):
            return node
        node = node.copy()

        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
                raise ValueError("MCP tool schemas only support local JSON references")
            if len(seen_refs) >= 64:
                raise ValueError("MCP tool schema reference depth exceeds 64")
            if ref in seen_refs:
                raise ValueError(
                    f"MCP tool schema has a recursive JSON reference: {ref}"
                )
            target = root
            try:
                for part in ref[2:].split("/") if ref != "#" else []:
                    key = part.replace("~1", "/").replace("~0", "~")
                    target = (
                        target[int(key)] if isinstance(target, list) else target[key]
                    )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Unresolved MCP tool schema reference: {ref}"
                ) from error
            if not isinstance(target, dict):
                raise ValueError(f"MCP tool schema reference is not an object: {ref}")
            node = {
                **target,
                **{key: value for key, value in node.items() if key != "$ref"},
            }
            return normalize(node, seen_refs + (ref,))

        node.pop("title", None)
        if "default" in node and node["default"] is None:
            del node["default"]
        if "anyOf" in node:
            variants = [normalize(variant, seen_refs) for variant in node.pop("anyOf")]
            variants = [
                variant for variant in variants if variant.get("type") != "null"
            ]
            if variants and all(
                set(variant) == {"type"} and isinstance(variant["type"], str)
                for variant in variants
            ):
                # Keep the existing shape for primitive unions.
                node["type"] = [variant["type"] for variant in variants]
            elif len(variants) == 1:
                node = {**variants[0], **node}
            elif variants:
                # Flattening these into type names loses enum values, object
                # properties and array items. Harmony renders oneOf as a union.
                node["oneOf"] = variants
            else:
                node["type"] = "null"
        if "properties" in node:
            node["properties"] = {
                key: normalize(value, seen_refs)
                for key, value in node["properties"].items()
            }
        if "items" in node:
            node["items"] = normalize(node["items"], seen_refs)
        if "oneOf" in node:
            node["oneOf"] = [normalize(value, seen_refs) for value in node["oneOf"]]
        return node

    return normalize(schema)


def post_process_tools_description(
    list_tools_result: "ListToolsResult",
) -> "ListToolsResult":
    # Adapt the MCP tool result for Harmony
    for tool in list_tools_result.tools:
        tool.inputSchema = trim_schema(tool.inputSchema)

    # Some tools schema don't need to be part of the prompt (e.g. simple text
    # in text out for Python)
    list_tools_result.tools = [
        tool
        for tool in list_tools_result.tools
        if getattr(tool.annotations, "include_in_prompt", True)
    ]

    return list_tools_result


class ToolServer(ABC):
    @abstractmethod
    def has_tool(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_description(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_session(self, tool_name: str) -> AbstractAsyncContextManager[Any]: ...


class MCPToolServer(ToolServer):
    def __init__(self):
        self.harmony_tool_descriptions = {}

    async def add_tool_server(self, server_url: str):
        tool_urls = server_url.split(",")
        self.harmony_tool_descriptions = {}
        self.urls: dict[str, str] = {}
        for url in tool_urls:
            url = f"http://{url}/sse"
            initialize_response, list_tools_response = await list_server_and_tools(url)

            list_tools_response = post_process_tools_description(list_tools_response)

            tool_from_mcp = ToolNamespaceConfig(
                name=initialize_response.serverInfo.name,
                description=initialize_response.instructions,
                tools=[
                    ToolDescription.new(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema,
                    )
                    for tool in list_tools_response.tools
                ],
            )
            self.harmony_tool_descriptions[tool_from_mcp.name] = tool_from_mcp
            if tool_from_mcp.name not in self.urls:
                self.urls[tool_from_mcp.name] = url
            else:
                logger.warning(
                    "Tool %s already exists. Ignoring duplicate tool server %s",
                    tool_from_mcp.name,
                    url,
                )

    def has_tool(self, tool_name: str):
        return tool_name in self.harmony_tool_descriptions

    def get_tool_description(self, tool_name: str):
        return self.harmony_tool_descriptions.get(tool_name)

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        url = self.urls.get(tool_name)
        if url:
            async with (
                sse_client(url=url) as streams,
                ClientSession(*streams) as session,
            ):
                await session.initialize()
                yield session
        else:
            logger.warning("Tool %s not found", tool_name)


class DemoToolServer(ToolServer):
    def __init__(self, *, enable_python: bool = True):
        from sglang.srt.entrypoints.tool import (
            HarmonyBrowserTool,
            HarmonyPythonTool,
            Tool,
        )

        self.tools: dict[str, Tool] = {}
        browser_tool = HarmonyBrowserTool()
        if browser_tool.enabled:
            self.tools["browser"] = browser_tool
        if enable_python:
            python_tool = HarmonyPythonTool()
            if python_tool.enabled:
                self.tools["python"] = python_tool

    def has_tool(self, tool_name: str):
        return tool_name in self.tools

    def get_tool_description(self, tool_name: str):
        if tool_name not in self.tools:
            return None
        if tool_name == "browser":
            return ToolNamespaceConfig.browser()
        elif tool_name == "python":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown tool {tool_name}")

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        yield self.tools[tool_name]

    async def aclose(self):
        browser = self.tools.get("browser")
        exa_client = getattr(browser, "exa_client", None) if browser else None
        if exa_client is not None:
            await exa_client.close()


class NativeToolServer(DemoToolServer):
    """Built-in SGLang hosted tools that do not require an external MCP server."""

    def __init__(self):
        super().__init__(enable_python=False)
