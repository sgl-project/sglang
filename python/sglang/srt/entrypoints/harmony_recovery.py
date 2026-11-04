# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow recovery for two observed malformed Harmony tool headers.

Adapted from vLLM PR #59387, commit 756ee24f369475638e4bb887828fa45333425864:
https://github.com/vllm-project/vllm/pull/59387
Valid input is passed directly to Harmony; recovery never rewrites prompt history.
"""

import re
from collections.abc import Collection

import orjson
from openai_harmony import (
    HarmonyEncoding,
    HarmonyError,
    Role,
    StreamableParser,
    StreamState,
)


class RecoveringHarmonyParser:
    """Keep Harmony's streaming interface while recovering recognized headers."""

    def __init__(
        self,
        encoding: HarmonyEncoding,
        function_names: Collection[str] | None = None,
        role: Role | None = Role.ASSISTANT,
    ):
        self._encoding = encoding
        self._role = role
        self._function_names = (
            None if function_names is None else frozenset(function_names)
        )
        self._parser = StreamableParser(encoding, role=role)
        self._completed_messages = []
        self._completed_tokens = []
        self._header_tokens = []
        self._header_start = 0
        self._num_parser_tokens = 0
        self._collect_header = role is not None
        self._start, self._channel, self._constrain, self._message, self._call = (
            encoding.encode(text, allowed_special={text}, disallowed_special=())[0]
            for text in (
                "<|start|>",
                "<|channel|>",
                "<|constrain|>",
                "<|message|>",
                "<|call|>",
            )
        )

    def __getattr__(self, name):
        return getattr(self._parser, name)

    @property
    def messages(self):
        return self._completed_messages + self._parser.messages

    @property
    def tokens(self):
        return self._completed_tokens + self._parser.tokens

    def process(self, token: int):
        if (
            token == self._start
            and not self._collect_header
            and self._parser.state == StreamState.EXPECT_START
        ):
            self._collect_header = True
            self._header_start = self._num_parser_tokens
        if self._collect_header:
            self._header_tokens.append(token)
        try:
            self._parser.process(token)
        except HarmonyError:
            if token != self._message or not self._repair_metadata():
                raise
        else:
            self._num_parser_tokens += 1
        if token == self._message and self._parser.state == StreamState.CONTENT:
            self._header_tokens.clear()
            self._collect_header = False
        return self

    def finish(self):
        # Partial content remains available to SGLang's existing finalization.
        if self._parser.state == StreamState.HEADER:
            try:
                self._parser.process_eos()
            except HarmonyError:
                self._repair_missing_message()
        return self

    def _assistant_prefix(self, tokens: list[int]) -> bool:
        if tokens and tokens[0] == self._start:
            prefix = self._encoding.decode(tokens[1:])
            pattern = r"assistant(?: to=[A-Za-z0-9_.-]+)?"
        else:
            if self._role != Role.ASSISTANT:
                return False
            prefix = self._encoding.decode(tokens)
            pattern = r"(?: to=[A-Za-z0-9_.-]+)?"
        return re.fullmatch(pattern, prefix) is not None

    def _replace_parser(self, tokens: list[int]) -> bool:
        role = None if tokens[0] == self._start else Role.ASSISTANT
        parser = StreamableParser(self._encoding, role=role)
        try:
            for token in tokens:
                parser.process(token)
        except HarmonyError:
            return False
        # Only materialize messages when replacing the parser, never per token.
        self._completed_messages.extend(self._parser.messages)
        self._completed_tokens.extend(self._parser.tokens[: self._header_start])
        self._parser = parser
        self._num_parser_tokens = len(tokens)
        self._header_start = 0
        self._header_tokens.clear()
        self._collect_header = False
        return True

    def _metadata_parts(self, tokens: list[int]):
        parts, plain = [], []
        duplicate_marker = False
        for token in tokens:
            if token in (self._channel, self._constrain):
                parts.extend(self._encoding.decode(plain).split())
                plain.clear()
                duplicate_marker = True
            elif self._encoding.is_special_token(token):
                return [], False
            else:
                plain.append(token)
        parts.extend(self._encoding.decode(plain).split())
        return parts, duplicate_marker

    def _repair_metadata(self) -> bool:
        tokens = self._header_tokens[:-1]
        try:
            channel = tokens.index(self._channel)
            constrain = tokens.index(self._constrain, channel + 1)
        except ValueError:
            return False
        if not self._assistant_prefix(tokens[:channel]):
            return False
        prefix = (
            self._encoding.decode(tokens[:channel])
            + " "
            + self._encoding.decode(tokens[channel + 1 : constrain])
        )
        recipients = re.findall(r"(?:^|\s)to=([A-Za-z0-9_.-]+)(?=\s|$)", prefix)
        if len(recipients) != 1:
            return False
        parts, duplicate_marker = self._metadata_parts(tokens[constrain + 1 :])
        if not parts or (len(parts) < 2 and not duplicate_marker):
            return False
        if parts[-1] not in ("code", "json"):
            return False
        if any(
            part not in ("analysis", "commentary", "final", recipients[0])
            and re.fullmatch(r"to=[A-Za-z0-9_.-]+", part) is None
            for part in parts[:-1]
        ):
            return False
        corrected = (
            tokens[: constrain + 1]
            + self._encoding.encode(parts[-1], disallowed_special=())
            + [self._message]
        )
        return self._replace_parser(corrected)

    def _argument_tokens(self, payload: list[int], boundary: int) -> list[int]:
        # Only the rare token straddling header/body needs retokenization.
        # All subsequent argument IDs, including literal control text, survive.
        for index in range(1, len(payload) + 1):
            try:
                prefix = self._encoding.decode_utf8(payload[:index])
            except HarmonyError:
                continue
            if len(prefix) >= boundary:
                fragment = prefix[boundary:]
                return (
                    self._encoding.encode(fragment, disallowed_special=())
                    + payload[index:]
                )
        return []

    def _missing_message_parts(self):
        tokens = self._header_tokens
        if not tokens or tokens[-1] != self._call:
            return None
        try:
            channel_index = tokens.index(self._channel)
        except ValueError:
            return None
        prefix = tokens[:channel_index]
        # This recovery only recognizes channel-first function headers.
        if prefix and prefix != self._encoding.encode(
            "<|start|>assistant", allowed_special={"<|start|>"}
        ):
            return None
        if not self._assistant_prefix(prefix):
            return None
        payload = tokens[channel_index + 1 : -1]
        argument_tokens = None
        if self._constrain in payload:
            index = payload.index(self._constrain)
            header = self._encoding.decode_utf8(payload[:index]).rstrip(" \t")
            if (
                re.fullmatch(
                    r"(?:analysis|commentary) to=functions\.[A-Za-z0-9_-]+", header
                )
                is None
            ):
                return None
            argument_tokens = payload[index + 1 :]
            payload = (
                self._encoding.encode(header, disallowed_special=()) + argument_tokens
            )
        if any(self._encoding.is_special_token(token) for token in payload):
            return None
        text = self._encoding.decode_utf8(payload)
        match = re.fullmatch(
            r"(analysis|commentary) to=(functions\.[A-Za-z0-9_-]+)"
            r"([ \t\r\n]*\{.*\}[ \t\r\n]*)",
            text,
            re.DOTALL,
        )
        if match is None:
            return None
        channel, recipient, arguments = match.groups()
        if (
            self._function_names is not None
            and recipient.removeprefix("functions.") not in self._function_names
        ):
            return None
        if not isinstance(orjson.loads(arguments), dict):
            return None
        if argument_tokens is None:
            argument_tokens = self._argument_tokens(payload, match.start(3))
        return channel, recipient, argument_tokens

    def _repair_missing_message(self) -> bool:
        try:
            parts = self._missing_message_parts()
        except (HarmonyError, ValueError, RecursionError):
            return False
        if parts is None:
            return False
        channel, recipient, arguments = parts
        header = self._encoding.encode(
            f"<|channel|>{channel} to={recipient} <|constrain|>json<|message|>",
            allowed_special={"<|channel|>", "<|constrain|>", "<|message|>"},
        )
        return self._replace_parser(header + arguments + [self._call])
