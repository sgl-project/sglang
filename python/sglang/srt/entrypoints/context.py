# SPDX-License-Identifier: Apache-2.0
# Copied from vLLM
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import orjson

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
except ImportError as e:
    mcp = e

from openai_harmony import Author, Message, Role, StreamState, TextContent

from sglang.srt.entrypoints.harmony_utils import (
    get_encoding,
    get_streamable_parser_for_assistant,
    render_for_completion,
)
from sglang.srt.entrypoints.tool import Tool


class ConversationContext(ABC):

    @abstractmethod
    def append_output(self, output) -> None:
        pass

    @abstractmethod
    async def call_tool(self) -> list[Message]:
        pass

    @abstractmethod
    def need_builtin_tool_call(self) -> bool:
        pass

    @abstractmethod
    def render_for_completion(self) -> list[int]:
        pass


class SimpleContext(ConversationContext):

    def __init__(self):
        self.last_output = None

    def append_output(self, output) -> None:
        self.last_output = output

    def need_builtin_tool_call(self) -> bool:
        return False

    async def call_tool(self) -> list[Message]:
        raise NotImplementedError("Should not be called.")

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")


class HarmonyContext(ConversationContext):

    def __init__(
        self,
        messages: list,
        tool_sessions: dict[str, Union["ClientSession", Tool]],
    ):
        # TODO: Remove the hack of Union[ClientSession, Tool] by using MCP
        # when demo.
        self._messages = messages
        self.tool_sessions = tool_sessions

        self.parser = get_streamable_parser_for_assistant()
        self.num_init_messages = len(messages)
        # TODO
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
        self.num_reasoning_tokens = 0
        self.finish_reason = None
        # Logprobs for the tokens that decode to the visible ``final``-channel
        # answer. Bucketed during append_output by tracking the parser's current
        # channel, so reasoning / structural tokens are excluded. Parallel lists:
        # one (logprob, token_id, token_text) per captured token, and (for
        # top_logprobs requests) one top-k list per captured token.
        self.final_token_logprobs: list = []
        self.final_top_logprobs: Optional[list] = None

    def append_output(self, output) -> None:
        if isinstance(output, dict) and "output_ids" in output:
            output_token_ids = output["output_ids"]
            meta_info = output["meta_info"]

            token_logprobs = None
            top_logprobs = None
            if isinstance(meta_info, dict):
                token_logprobs = meta_info.get("output_token_logprobs")
                top_logprobs = meta_info.get("output_top_logprobs")

            for i, token_id in enumerate(output_token_ids):
                self.parser.process(token_id)
                # Bucket logprobs for the visible answer: only tokens the parser
                # attributes to the ``final`` channel AND that emit text.
                # Structural tokens (headers, <|message|>, <|return|>) are
                # excluded because last_content_delta is empty for them.
                if (
                    token_logprobs is not None
                    and self.parser.current_channel == "final"
                    and self.parser.last_content_delta
                ):
                    self.final_token_logprobs.append(token_logprobs[i])
                    if top_logprobs is not None:
                        if self.final_top_logprobs is None:
                            self.final_top_logprobs = []
                        self.final_top_logprobs.append(
                            top_logprobs[i] if i < len(top_logprobs) else None
                        )

            output_msgs = self.parser.messages

            if isinstance(meta_info, dict):
                if "prompt_token_ids" in meta_info:
                    self.num_prompt_tokens = meta_info["prompt_tokens"]
                if "cached_tokens" in meta_info:
                    self.num_cached_tokens = meta_info["cached_tokens"]
                if "completion_tokens" in meta_info:
                    self.num_output_tokens += meta_info["completion_tokens"]
                self._record_finish_reason(meta_info)

        else:
            output_msgs = output

        self._messages.extend(output_msgs)

    def _record_finish_reason(self, meta_info: dict) -> None:
        # Last non-null wins: a builtin-tool continuation turn supersedes the
        # reason recorded for the turn before it.
        reason = meta_info.get("finish_reason")
        if reason is not None:
            self.finish_reason = reason

    @property
    def messages(self) -> list:
        return self._messages

    def need_builtin_tool_call(self) -> bool:
        if not self.messages:
            return False
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        return recipient is not None and (
            recipient.startswith("browser.") or recipient.startswith("python")
        )

    async def call_tool(self) -> list[Message]:
        if not self.messages:
            return []
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is not None:
            if recipient.startswith("browser."):
                return await self.call_search_tool(
                    self.tool_sessions["browser"], last_msg
                )
            elif recipient.startswith("python"):
                return await self.call_python_tool(
                    self.tool_sessions["python"], last_msg
                )
        raise ValueError("No tool call found")

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)

    async def call_search_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        tool_name = last_msg.recipient.split(".")[1]
        args = orjson.loads(last_msg.content[0].text)
        result = await tool_session.call_tool(tool_name, args)
        result_str = result.content[0].text
        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name=last_msg.recipient)
        return [Message(author=author, content=[content], recipient=Role.ASSISTANT)]

    async def call_python_tool(
        self, tool_session: Union["ClientSession", Tool], last_msg: Message
    ) -> list[Message]:
        if isinstance(tool_session, Tool):
            return await tool_session.get_result(self)
        param = {
            "code": last_msg.content[0].text,
        }
        result = await tool_session.call_tool("python", param)
        result_str = result.content[0].text

        content = TextContent(text=result_str)
        author = Author(role=Role.TOOL, name="python")

        return [
            Message(
                author=author,
                content=[content],
                channel=last_msg.channel,
                recipient=Role.ASSISTANT,
            )
        ]


class StreamingHarmonyContext(HarmonyContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()
        self.encoding = get_encoding()
        self.last_tok = None
        self.num_processed_tokens = 0
        # Logprob of the most recent final-channel content token in the current
        # chunk, exposed so the streaming delta event can carry it. Reset every
        # append_output call; None when the chunk added no final-content token.
        self.last_delta_logprob = None
        self.last_delta_top_logprob = None

    @property
    def messages(self) -> list:
        return self.parser.messages

    def append_output(self, output) -> None:
        if isinstance(output, dict) and "output_ids" in output:
            # RequestOutput from SGLang with outputs
            output_token_ids = output["output_ids"]

            # Check if we need to handle cumulative tokens
            meta_info = output.get("meta_info", {})
            completion_tokens = meta_info.get("completion_tokens")
            all_token_logprobs = meta_info.get("output_token_logprobs")
            all_top_logprobs = meta_info.get("output_top_logprobs")
            if (
                completion_tokens is not None
                and len(output_token_ids) == completion_tokens
            ):
                # Case 1: When --incremental-streaming-output is not set.
                # The output_ids contains all tokens generated so far.
                # We only need to process the new tokens.
                new_token_ids = output_token_ids[self.num_processed_tokens :]
                new_token_logprobs = (
                    all_token_logprobs[self.num_processed_tokens :]
                    if all_token_logprobs is not None
                    else None
                )
                new_top_logprobs = (
                    all_top_logprobs[self.num_processed_tokens :]
                    if all_top_logprobs is not None
                    else None
                )
                self.num_processed_tokens = len(output_token_ids)
            else:
                # Case 2: When --incremental-streaming-output is set.
                # The output_ids contains only the new tokens.
                new_token_ids = output_token_ids
                new_token_logprobs = all_token_logprobs
                new_top_logprobs = all_top_logprobs
                self.num_processed_tokens += len(output_token_ids)

            self._record_finish_reason(meta_info)

            # Reset per-chunk delta state so last_delta_logprob always describes
            # the current chunk's last final-channel content token (or None).
            self.last_delta_logprob = None
            self.last_delta_top_logprob = None
            for i, token_id in enumerate(new_token_ids):
                self.parser.process(token_id)
                # Bucket final-channel content tokens (see HarmonyContext) and
                # track the latest one for the streaming delta event.
                if (
                    new_token_logprobs is not None
                    and self.parser.current_channel == "final"
                    and self.parser.last_content_delta
                ):
                    token_lp = new_token_logprobs[i]
                    self.final_token_logprobs.append(token_lp)
                    self.last_delta_logprob = token_lp
                    if new_top_logprobs is not None:
                        top_lp = (
                            new_top_logprobs[i]
                            if i < len(new_top_logprobs)
                            else None
                        )
                        if self.final_top_logprobs is None:
                            self.final_top_logprobs = []
                        self.final_top_logprobs.append(top_lp)
                        self.last_delta_top_logprob = top_lp

        else:
            # Handle the case of tool output in direct message format
            assert len(output) == 1, "Tool output should be a single message"
            msg = output[0]
            # Sometimes the recipient is not set for tool messages,
            # so we set it to "assistant"
            if msg.author.role == Role.TOOL and msg.recipient is None:
                msg.recipient = "assistant"
            toks = self.encoding.render(msg)
            for tok in toks:
                self.parser.process(tok)
            self.last_tok = toks[-1]

    def is_expecting_start(self) -> bool:
        return self.parser.state == StreamState.EXPECT_START

    def is_assistant_action_turn(self) -> bool:
        return self.last_tok in self.encoding.stop_tokens_for_assistant_actions()

    def render_for_completion(self) -> list[int]:
        # now this list of tokens as next turn's starting tokens
        # `<|start|>assistant``,
        # we need to process them in parser.
        rendered_tokens = super().render_for_completion()

        last_n = -1
        to_process = []
        while rendered_tokens[last_n] != self.last_tok:
            to_process.append(rendered_tokens[last_n])
            last_n -= 1
        for tok in reversed(to_process):
            self.parser.process(tok)

        return rendered_tokens
