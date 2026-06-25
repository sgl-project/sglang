# SPDX-License-Identifier: Apache-2.0
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import orjson

from sglang.srt.entrypoints.search.exa_client import (
    ExaClient,
    ExaSearchConfig,
)
from sglang.srt.utils import print_info_once, print_warning_once

if TYPE_CHECKING:
    # Avoid circular import.
    from sglang.srt.entrypoints.context import ConversationContext

logger = logging.getLogger(__name__)


class Tool(ABC):

    @abstractmethod
    async def get_result(self, context: "ConversationContext") -> Any:
        pass


class HarmonyBrowserTool(Tool):

    def __init__(self, client: ExaClient | None = None):
        self.enabled = True
        if client is not None:
            self.exa_client = client
            print_info_once("Browser tool initialized")
            return

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            self.enabled = False
            print_warning_once("EXA_API_KEY is not set, browsing is disabled")
            return

        self.exa_client = ExaClient(api_key, config=ExaSearchConfig.from_env())
        print_info_once("Browser tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from openai_harmony import Author, Message, Role, TextContent

        last_msg = context.messages[-1]
        recipient = last_msg.recipient
        if recipient is None or not recipient.startswith("browser."):
            raise ValueError("No browser tool call found")

        try:
            args = orjson.loads(last_msg.content[0].text)
            result_text = await self._dispatch_browser_call(context, recipient, args)
        except Exception as exc:
            logger.exception("Browser tool call failed")
            result_text = f"Browser tool call failed: {exc}"

        content = TextContent(text=result_text)
        author = Author(role=Role.TOOL, name=recipient)
        return [Message(author=author, content=[content], recipient=Role.ASSISTANT)]

    @property
    def tool_config(self) -> Any:
        return None

    async def _dispatch_browser_call(
        self, context: "ConversationContext", recipient: str, args: dict[str, Any]
    ) -> str:
        if recipient == "browser.search":
            query = args.get("query")
            if not query:
                raise ValueError("browser.search requires a query")
            data = await self.exa_client.search(query)
            return self._format_search_results(context, query, data)

        if recipient == "browser.open":
            url = self._resolve_url(context, args)
            data = await self.exa_client.contents([url])
            return self._format_page_contents(context, url, data)

        if recipient == "browser.find":
            pattern = args.get("pattern")
            if not pattern:
                raise ValueError("browser.find requires a pattern")
            return await self._find_pattern(context, args, pattern)

        raise ValueError(f"Unknown browser action: {recipient}")

    def _browser_state(self, context: "ConversationContext") -> dict[str, Any]:
        state = getattr(context, "_sglang_exa_browser_state", None)
        if state is None:
            state = {"pages": {}, "page_text": {}}
            setattr(context, "_sglang_exa_browser_state", state)
        return state

    def _format_search_results(
        self, context: "ConversationContext", query: str, data: dict[str, Any]
    ) -> str:
        state = self._browser_state(context)
        state["pages"] = {}
        state["page_text"] = {}

        results = data.get("results") or []
        request_id = data.get("requestId")
        if request_id:
            logger.debug("Exa search request id: %s", request_id)
        lines = [f"Search results for: {query}"]
        lines.append("Use browser.open with the cursor number to inspect a result.")

        for index, result in enumerate(results, start=1):
            cursor = str(index)
            state["pages"][cursor] = result
            title = result.get("title") or "Untitled"
            url = result.get("url") or result.get("id") or ""
            snippet = self._best_snippet(result)
            lines.append("")
            lines.append(f"[{cursor}] {title}")
            if url:
                lines.append(f"URL: {url}")
            if snippet:
                lines.append(f"Snippet: {snippet}")

        if not results:
            lines.append("No results found.")
        return "\n".join(lines)

    def _format_page_contents(
        self, context: "ConversationContext", url: str, data: dict[str, Any]
    ) -> str:
        state = self._browser_state(context)
        results = data.get("results") or []
        if not results:
            return f"No page contents returned for {url}."

        page = results[0]
        cursor = self._cursor_for_url(state, url)
        if cursor:
            state["pages"][cursor] = page
        title = page.get("title") or "Untitled"
        page_url = page.get("url") or url
        text = page.get("text") or self._best_snippet(page) or page.get("summary") or ""
        if cursor:
            state["page_text"][cursor] = text

        return "\n".join(
            [
                f"Opened page: {title}",
                f"URL: {page_url}",
                "",
                self._truncate(text, 12000) if text else "No page text available.",
            ]
        )

    async def _find_pattern(
        self, context: "ConversationContext", args: dict[str, Any], pattern: str
    ) -> str:
        state = self._browser_state(context)
        cursor = self._normalize_cursor(args.get("cursor"))

        if cursor:
            text = state["page_text"].get(cursor)
            if text is None:
                url = self._resolve_url(context, args)
                data = await self.exa_client.contents([url])
                self._format_page_contents(context, url, data)
                text = state["page_text"].get(cursor, "")
            return self._format_matches(pattern, text)

        if args.get("url"):
            url = self._resolve_url(context, args)
            data = await self.exa_client.contents([url])
            results = data.get("results") or []
            if not results:
                return f"No page contents returned for {url}."
            page = results[0]
            text = page.get("text") or self._best_snippet(page) or page.get("summary") or ""
            return self._format_matches(pattern, text)

        searchable_text = "\n\n".join(
            self._best_snippet(page) for page in state["pages"].values()
        )
        return self._format_matches(pattern, searchable_text)

    def _resolve_url(self, context: "ConversationContext", args: dict[str, Any]) -> str:
        if args.get("url"):
            return str(args["url"])

        state = self._browser_state(context)
        cursor = self._normalize_cursor(args.get("cursor"))
        if not cursor:
            raise ValueError("browser.open requires a cursor or url")

        page = state["pages"].get(cursor)
        if page is None:
            raise ValueError(f"Unknown browser cursor: {cursor}")

        url = page.get("url") or page.get("id")
        if not url:
            raise ValueError(f"No URL recorded for browser cursor: {cursor}")
        return str(url)

    def _normalize_cursor(self, cursor: Any) -> str | None:
        if cursor is None:
            return None
        cursor_str = str(cursor)
        if cursor_str == "0":
            return "1"
        return cursor_str

    def _cursor_for_url(self, state: dict[str, Any], url: str) -> str | None:
        for cursor, page in state["pages"].items():
            if page.get("url") == url or page.get("id") == url:
                return cursor
        return None

    def _best_snippet(self, result: dict[str, Any]) -> str:
        highlights = result.get("highlights") or []
        if highlights:
            return self._truncate(str(highlights[0]), 1000)
        summary = result.get("summary")
        if summary:
            return self._truncate(str(summary), 1000)
        text = result.get("text")
        if text:
            return self._truncate(str(text), 1000)
        return ""

    def _format_matches(self, pattern: str, text: str) -> str:
        if not text:
            return f"No text available to search for {pattern!r}."

        pattern_lower = pattern.lower()
        matches = []
        for line in text.splitlines():
            if pattern_lower in line.lower():
                matches.append(self._truncate(line.strip(), 1000))
            if len(matches) >= 10:
                break

        if not matches:
            return f"No matches found for {pattern!r}."

        lines = [f"Matches for {pattern!r}:"]
        lines.extend(f"- {match}" for match in matches)
        return "\n".join(lines)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."


class HarmonyPythonTool(Tool):

    def __init__(self):
        self.enabled = True

        try:
            from gpt_oss.tools.python_docker.docker_tool import PythonTool
        except ImportError:
            self.enabled = False
            print_warning_once("gpt_oss is not installed, code interpreter is disabled")
            return

        self.python_tool = PythonTool()
        print_info_once("Code interpreter tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from sglang.srt.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.python_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        return self.python_tool.tool_config
