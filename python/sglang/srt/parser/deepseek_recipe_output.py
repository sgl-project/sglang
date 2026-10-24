"""Token-level output parsing for DeepSeek-V4 / V4.1 through deepseek-recipe's
StreamProcessor: `</think>` and the DSML tags are recognized by token ID, as the API
parses them, and tool-call arguments arrive as the API streams them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Optional

import msgspec

from sglang.srt.parser.deepseek_recipe_renderer import (
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    DeepSeekRecipe,
    load_deepseek_recipe,
)


class RecipeToolCallDelta(msgspec.Struct, frozen=True):
    """A piece of one call's JSON arguments; the first piece also names the call."""

    index: int
    arguments: str
    name: Optional[str] = None


class RecipeOutputDelta(msgspec.Struct, frozen=True):
    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[RecipeToolCallDelta, ...] = ()

    def merge(self, other: RecipeOutputDelta) -> RecipeOutputDelta:
        return RecipeOutputDelta(
            reasoning=self.reasoning + other.reasoning,
            content=self.content + other.content,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class RecipeToolCall(msgspec.Struct, frozen=True):
    index: int
    name: str
    arguments: str


def whole_tool_calls(delta: RecipeOutputDelta) -> list[RecipeToolCall]:
    calls: dict[int, list[str]] = {}
    for piece in delta.tool_calls:
        if piece.name is not None:
            calls[piece.index] = [piece.name, piece.arguments]
        else:
            calls[piece.index][1] += piece.arguments
    return [
        RecipeToolCall(index=index, name=name, arguments=arguments)
        for index, (name, arguments) in sorted(calls.items())
    ]


class DeepSeekRecipeOutputParser:
    """One choice's output through recipe's StreamProcessor, fed the engine's token
    ids. Stops stay SGLang's: the scheduler matches them and `finish` cuts the match.
    `thinking_stage` is recipe's ThinkingInitialStage, None after a non-thinking
    prompt."""

    def __init__(
        self,
        deepseek_recipe: DeepSeekRecipe,
        *,
        thinking_stage: Any,
        parse_tool_calls: bool,
        parse_json_output: bool = False,
        separate_reasoning: bool = True,
        stream_reasoning: bool = True,
        stop: Sequence[str] = (),
    ):
        recipe = load_deepseek_recipe()
        options = recipe.ParsingOptions(
            parse_tool_calls=parse_tool_calls,
            parse_json_output=parse_json_output,
            thinking_initial_stage=thinking_stage,
            special_tokens=deepseek_recipe.special_tokens,
        )
        # The generator only frames chunks; what is parsed follows `options`.
        request = recipe.ConversationRequest(
            recipe.Conversation(), deepseek_recipe.model_version
        )
        generator = recipe.ChatCompletionRequest.chunk_generator(
            request, "", deepseek_recipe.model
        )
        self._recipe = recipe
        self._processor = recipe.StreamProcessor(
            generator, options, deepseek_recipe.tokenizer
        )
        self._hf_tokenizer = deepseek_recipe.hf_tokenizer
        self._reasoning_as_content = not separate_reasoning
        self._hold_reasoning = separate_reasoning and not stream_reasoning
        self._held_reasoning = ""
        # Ids a stop string may still start in wait until the scheduler rules it
        # out, so a matched stop is cut before recipe has emitted any of it.
        self._stop_hold_chars = max(map(len, stop), default=1) - 1
        self._pending_ids: list[int] = []
        self.num_consumed_tokens = 0

    def feed(self, token_ids: Sequence[int]) -> RecipeOutputDelta:
        self.num_consumed_tokens += len(token_ids)
        if self._processor.finished:
            return RecipeOutputDelta()
        self._pending_ids += token_ids
        ready = self._release_pending_ids()
        if not ready:
            return RecipeOutputDelta()
        return self._deltas(
            self._processor.push(self._recipe.InferenceChunk.tokens(ready))
        )

    def _release_pending_ids(self) -> list[int]:
        held = len(self._pending_ids)
        while held > 0 and self._stop_hold_chars > len(
            self._hf_tokenizer.decode(
                self._pending_ids[held:], skip_special_tokens=False
            )
        ):
            held -= 1
        ready, self._pending_ids = self._pending_ids[:held], self._pending_ids[held:]
        return ready

    def finish(
        self,
        token_ids: Sequence[int],
        finish_reason: dict[str, Any],
        *,
        keep_matched_stop: bool = False,
    ) -> RecipeOutputDelta:
        """The last engine chunk: its ids, less the matched stop, and its reason."""
        self.num_consumed_tokens += len(token_ids)
        if self._processor.finished:
            return RecipeOutputDelta()  # a sticky finish_reason re-sent
        token_ids = self._pending_ids + list(token_ids)
        self._pending_ids = []
        matched = finish_reason.get("matched")
        if isinstance(matched, str):
            token_ids = self._cut_at_stop_string(
                token_ids, matched, keep_stop=keep_matched_stop
            )
        elif (
            isinstance(matched, int)
            and not keep_matched_stop
            and token_ids
            and token_ids[-1] == matched
        ):
            token_ids = token_ids[:-1]
        chunks = (
            self._processor.push(self._recipe.InferenceChunk.tokens(token_ids))
            if token_ids
            else []
        )
        if not self._processor.finished:
            reason = {
                "stop": self._recipe.InferenceFinishReason.Stop,
                "length": self._recipe.InferenceFinishReason.Length,
            }.get(finish_reason["type"])
            if reason is None:  # abort: flush the parse without a backend finish
                chunks += self._processor.finish()
            else:
                chunks += self._processor.push(
                    self._recipe.InferenceChunk.finish(reason)
                )
        return self._deltas(chunks)

    def _cut_at_stop_string(
        self, token_ids: list[int], stop: str, *, keep_stop: bool
    ) -> list[int]:
        # The scheduler matched `stop` in the decoded text and ended on the token
        # completing it; the earliest match ending there starts at most
        # len(stop) - 1 characters before that token.
        decode = self._hf_tokenizer.decode
        start = len(token_ids) - 1
        last = decode(token_ids[start:], skip_special_tokens=False)
        text = last
        while start > 0 and (stop not in text or len(text) - len(last) < len(stop) - 1):
            start -= 1
            text = decode(token_ids[start:], skip_special_tokens=False)
        cut = text.find(stop)
        if cut == -1:
            return token_ids
        if keep_stop:
            cut += len(stop)
        return (
            token_ids[:start]
            + self._hf_tokenizer.encode(text[:cut], add_special_tokens=False).ids
        )

    def _deltas(self, chunks: list[Any]) -> RecipeOutputDelta:
        delta = RecipeOutputDelta()
        for chunk in chunks:
            choice = json.loads(chunk.to_json())["choices"][0]
            piece = choice["delta"]
            if piece.get("role"):  # the handler emits its own role chunk
                continue
            reasoning = piece.get("reasoning_content") or ""
            content = piece.get("content") or ""
            tool_calls = tuple(
                RecipeToolCallDelta(
                    index=call["index"],
                    arguments=call["function"]["arguments"],
                    name=call["function"].get("name"),
                )
                for call in piece.get("tool_calls") or ()
            )
            if self._reasoning_as_content:
                content, reasoning = reasoning + content, ""
            if self._hold_reasoning:
                self._held_reasoning += reasoning
                reasoning = ""
                if content or tool_calls or choice["finish_reason"]:
                    reasoning, self._held_reasoning = self._held_reasoning, ""
            delta = delta.merge(
                RecipeOutputDelta(
                    reasoning=reasoning, content=content, tool_calls=tool_calls
                )
            )
        return delta


def parse_output_text(
    deepseek_recipe: DeepSeekRecipe,
    text: str,
    *,
    parse_tool_calls: bool,
    thinking: Optional[bool] = None,
) -> RecipeOutputDelta:
    """`text` parsed as a whole output; the markers it spells count as the tokens
    the model would have sampled. With `thinking` None, text that opens with
    `<think>` or closes a thinking block is reasoning up to `</think>`."""
    tokenizer = deepseek_recipe.hf_tokenizer
    output_ids = tokenizer.encode(text, add_special_tokens=False).ids
    if thinking is None:
        thinking = output_ids[:1] == [tokenizer.token_to_id(THINK_START_TOKEN)]
        if thinking:
            output_ids = output_ids[1:]
        thinking = thinking or tokenizer.token_to_id(THINK_END_TOKEN) in output_ids
    parser = DeepSeekRecipeOutputParser(
        deepseek_recipe,
        thinking_stage=deepseek_recipe.parse_start(thinking=thinking),
        parse_tool_calls=parse_tool_calls,
    )
    return parser.finish(output_ids, {"type": "stop"})
