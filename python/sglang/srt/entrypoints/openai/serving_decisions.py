from __future__ import annotations

import asyncio
import json
import logging
import math
import string
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import ORJSONResponse
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.openai.protocol import (
    DecisionAnswer,
    DecisionChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionScoreQuestion,
    DecisionText,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import _CHAT_TEMPLATE_CLIENT_ERRORS
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Version of the server-owned prompt wording and answer labels.
# Any change to _render_question or _answer_labels needs a new version.
PROMPT_FORMAT_VERSION = 1

# Parser defaults that name the chat template kwarg toggling reasoning.
_PARSER_TOGGLE_MODES = (
    "thinking",
    "enable_thinking",
    "explicit_thinking",
    "explicit_enable_thinking",
)
# Answer text of a finished reply, rendered only to see what precedes an answer.
_REPLY_SENTINEL = "DECISION_ANSWER"


class OpenAIServingDecisions(OpenAIServingBase):
    """Handler for /v1/decisions requests, answered by candidate scoring without generation"""

    def __init__(self, chat_serving: OpenAIServingChat):
        super().__init__(chat_serving.tokenizer_manager)
        # Render the way the chat route does, and refuse where it renders differently.
        self.template_manager = chat_serving.template_manager
        self.default_chat_template_kwargs = chat_serving.default_chat_template_kwargs
        self.chat_encoding_spec = chat_serving.chat_encoding_spec
        self.prompt_text_is_lossy = chat_serving._prompt_text_round_trip_is_lossy
        tokenizer = self.tokenizer_manager.tokenizer
        # Other tokenizers skip the shortcut in _encode_labels and check the full prompt.
        self.added_tokens = (
            {i: token for token, i in tokenizer.get_added_vocab().items()}
            if isinstance(tokenizer, PreTrainedTokenizerBase)
            else {}
        )
        # The configured reasoning parser, or the one the chat template suggests,
        # tells where reasoning blocks start and end and whether answers open one.
        parser = (
            chat_serving.reasoning_parser
            or self.template_manager.suggested_reasoning_parser
        )
        self.reasoning_markers = None
        self.answers_open_reasoning = False
        # The kwarg that turns reasoning on and off: the template's, else, when
        # detection finds no config, the one the configured or suggested parser names.
        config = self.template_manager.reasoning_config
        self.reasoning_toggle = config.toggle_param if config is not None else None
        if parser is not None:
            try:
                detector = ReasoningParser(
                    model_type=parser, tokenizer=tokenizer
                ).detector
            except ValueError as e:
                logger.warning(
                    "No reasoning block check for /v1/decisions with parser '%s': %s",
                    parser,
                    e,
                )
            else:
                mode = detector.reasoning_default
                if config is None and mode in _PARSER_TOGGLE_MODES:
                    self.reasoning_toggle = mode.removeprefix("explicit_")
                if detector.think_start_token and detector.think_end_token:
                    self.reasoning_markers = (
                        detector.think_start_token,
                        detector.think_end_token,
                    )
                    self.answers_open_reasoning = detector.reasoning_default == "always"

    def _request_id_prefix(self) -> str:
        return "decision-"

    def _validate_request(self, request: DecisionRequest) -> Optional[str]:
        if not self.tokenizer_manager.is_generation:
            return "/v1/decisions requires a generation model"
        if self.tokenizer_manager.tokenizer is None:
            return "/v1/decisions requires the server tokenizer"
        if self.chat_encoding_spec is not None:
            return (
                "/v1/decisions requires a chat template, but this model's chat "
                f"route uses the {self.chat_encoding_spec!r} encoder"
            )
        if self.prompt_text_is_lossy:
            return (
                "/v1/decisions places answer labels on the rendered chat text, "
                "which this tokenizer does not encode back to the same ids"
            )
        if self.template_manager.chat_template_name is not None:
            return (
                "/v1/decisions renders the tokenizer's Jinja chat template, but "
                "this server uses the built-in chat template "
                f"{self.template_manager.chat_template_name!r}"
            )
        if get_exec().features.enable_mis:
            return "/v1/decisions does not support --enable-mis"
        if get_exec().dllm.dllm_algorithm is not None:
            return (
                "/v1/decisions does not support diffusion language models "
                "served with --dllm-algorithm"
            )
        _, adapter = self._parse_model_parameter(request.model)
        if adapter is not None:
            return (
                f"model names the LoRA adapter {adapter!r}, which /v1/decisions "
                "does not support"
            )
        version = request.prompt_format_version
        if version is not None and version != PROMPT_FORMAT_VERSION:
            return (
                f"prompt_format_version {version} is not served, this server "
                f"uses version {PROMPT_FORMAT_VERSION}"
            )
        # The answer position must follow the reasoning block, not sit inside it.
        config = self.template_manager.reasoning_config
        if config is not None and config.always_on:
            return (
                "/v1/decisions does not support chat templates that always "
                "reason before answering"
            )
        toggle = self.reasoning_toggle
        kwargs = request.chat_template_kwargs
        if toggle in kwargs and kwargs[toggle] is not False:
            return (
                f"chat_template_kwargs sets {toggle!r} to {kwargs[toggle]!r}, "
                "but decisions need it false or unset"
            )
        return None

    def _chat_template_kwargs(self, request: DecisionRequest) -> Dict[str, Any]:
        """Reasoning off, then the server defaults, then the request kwargs."""
        kwargs = {}
        if self.reasoning_toggle is not None:
            kwargs[self.reasoning_toggle] = False
        for key, value in self.default_chat_template_kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.update(request.chat_template_kwargs)
        return kwargs

    def _convert_to_internal_request(
        self,
        request: DecisionRequest,
        raw_request: Request = None,
    ) -> Tuple[Iterator[Tuple[List[int], List[int]]], DecisionRequest]:
        # Lazy, so the handler can let other requests run between questions.
        return self._encoded_questions(request), request

    def _encoded_questions(
        self, request: DecisionRequest
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Prompt and label ids for each question, in request order."""
        text = _render_text(request.input)
        chat_template_kwargs = self._chat_template_kwargs(request)
        for question in request.questions:
            try:
                encoded = self._encode_question(
                    text=text,
                    question=question,
                    chat_template_kwargs=chat_template_kwargs,
                )
            except ValueError as e:
                raise ValueError(f"question {question.id!r}: {e}") from e
            yield encoded

    def _encode_question(
        self,
        text: str,
        question: DecisionQuestion,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        tokenizer = self.tokenizer_manager.tokenizer
        _, labels = _answer_labels(question)
        content = _render_question(text=text, question=question, labels=labels)
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS as e:
            raise ValueError(f"the chat template failed: {e}") from e
        if self.reasoning_markers is not None:
            # Look only after the message, whose last line is fixed text.
            closing = content.rsplit("\n", 1)[-1]
            cut = prompt.rfind(closing)
            generation_prompt = prompt if cut < 0 else prompt[cut + len(closing) :]
            start, end = self.reasoning_markers
            opened = generation_prompt.rfind(start)
            closed = generation_prompt.rfind(end)
            if opened > closed:
                raise ValueError(
                    "the chat template leaves a reasoning block open at the "
                    "answer position, so this model is not supported with these "
                    "chat_template_kwargs"
                )
            if self.answers_open_reasoning and closed < 0:
                raise ValueError(
                    "the reasoning parser for this model expects answers to start "
                    "with a reasoning block, and the chat template does not close "
                    "one. Send chat_template_kwargs that turn thinking off, if the "
                    "template supports it"
                )
            # The template's own finished reply shows whether answers start with
            # a reasoning block that the generation prompt leaves out.
            reply = self._render_reply(closing, chat_template_kwargs)
            begin = reply.rfind(closing) if reply is not None else -1
            answer = reply.find(_REPLY_SENTINEL, begin) if begin >= 0 else -1
            if answer >= 0 and reply[begin:answer].count(start) > (
                generation_prompt.count(start)
            ):
                raise ValueError(
                    "the chat template starts every answer with a reasoning "
                    "block, so this model is not supported"
                )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Refuse here because --allow-auto-truncate would cut off the answer position.
        context_len = self.tokenizer_manager.context_len
        if len(prompt_ids) + self.tokenizer_manager.num_reserved_tokens >= context_len:
            raise ValueError(
                f"the prompt has {len(prompt_ids)} tokens, which does not fit "
                f"the context length of {context_len} tokens"
            )
        label_ids = _encode_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            labels=labels,
            added_tokens=self.added_tokens,
        )
        return prompt_ids, label_ids

    def _render_reply(
        self, message: str, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": _REPLY_SENTINEL},
                ],
                tokenize=False,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS:
            # The generation prompt checks above still apply.
            return None

    async def _handle_non_streaming_request(
        self,
        adapted_request: Iterator[Tuple[List[int], List[int]]],
        request: DecisionRequest,
        raw_request: Request,
    ) -> ORJSONResponse:
        prompts, label_token_ids = [], []
        for prompt_ids, label_ids in adapted_request:
            prompts.append(prompt_ids)
            label_token_ids.append(label_ids)
            # Each question renders and tokenizes the whole input on the event loop.
            await asyncio.sleep(0)
        result = await self.tokenizer_manager.score_prompts(
            prompts=prompts,
            label_token_ids=label_token_ids,
            apply_softmax=True,
            request=raw_request,
            temperature=request.temperature,
            return_token_logprobs=True,
        )
        answers = {}
        for i, question in enumerate(request.questions):
            answer = _build_answer(
                question=question,
                probabilities=result.scores[i],
                token_logprobs=result.token_logprobs[i],
            )
            if request.return_prompt_token_ids:
                answer.prompt_token_ids = prompts[i]
                answer.label_token_ids = label_token_ids[i]
            answers[question.id] = answer
        response = DecisionResponse(
            model=request.model,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            answers=answers,
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.prompt_tokens,
            ),
        )
        return ORJSONResponse(content=response.model_dump(exclude_none=True))


def _render_text(value: Optional[DecisionText]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _answer_labels(question: DecisionQuestion) -> Tuple[List[str], List[str]]:
    """Answer names in the response and their single-token labels, in candidate order."""
    if isinstance(question, DecisionChoiceQuestion):
        names = [option.name for option in question.options]
        return names, list(string.ascii_uppercase[: len(names)])
    if isinstance(question, DecisionScoreQuestion):
        levels = [str(level) for level in range(len(question.levels))]
        return levels, levels
    return ["yes", "no"], ["yes", "no"]


def _render_question(text: str, question: DecisionQuestion, labels: List[str]) -> str:
    """Prompt wording of PROMPT_FORMAT_VERSION."""
    question_text = _render_text(question.question)
    if isinstance(question, DecisionChoiceQuestion):
        lines = [f"Question: {question_text}"]
        for label, option in zip(labels, question.options):
            detail = _render_text(option.description)
            lines.append(
                f"{label}: {option.name} - {detail}"
                if detail
                else f"{label}: {option.name}"
            )
        lines.append("Answer with the letter of one option only.")
    elif isinstance(question, DecisionScoreQuestion):
        lines = [f"Question: {question_text}"]
        lines += [
            f"{label}: {_render_text(level)}"
            for label, level in zip(labels, question.levels)
        ]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [f"Is the following true? {question_text}"]
        for label, description in zip(labels, (question.yes, question.no)):
            detail = _render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


def _encode_labels(
    tokenizer: Any,
    prompt: str,
    prompt_ids: List[int],
    labels: List[str],
    added_tokens: Dict[int, str],
) -> List[int]:
    """Check that each label adds exactly one distinct token after the prompt."""
    # Added tokens are split off before tokenization.
    # The text after the last one tokenizes on its own,
    # so the check does not grow with the input.
    # When the prompt ends with an added token, each label starts a new segment,
    # which is how the model continues after that token.
    text, text_ids = prompt, prompt_ids
    last = next(
        (i for i in reversed(range(len(prompt_ids))) if prompt_ids[i] in added_tokens),
        None,
    )
    if last is not None:
        token = added_tokens[prompt_ids[last]]
        start = prompt.rfind(token)
        suffix = prompt[start + len(token) :]
        if start >= 0 and (
            tokenizer.encode(suffix, add_special_tokens=False) == prompt_ids[last + 1 :]
        ):
            text, text_ids = suffix, prompt_ids[last + 1 :]
    label_ids = []
    for label in labels:
        ids = tokenizer.encode(text + label, add_special_tokens=False)
        if (
            len(ids) != len(text_ids) + 1
            or ids[:-1] != text_ids
            or ids[-1] in label_ids
        ):
            raise ValueError(
                f"the answer label {label!r} is not one distinct token after the "
                "chat prompt for this tokenizer, so this model is not supported"
            )
        label_ids.append(ids[-1])
    return label_ids


def _build_answer(
    question: DecisionQuestion,
    probabilities: List[float],
    token_logprobs: List[float],
) -> DecisionAnswer:
    names, _ = _answer_labels(question)
    value = {}
    if isinstance(question, DecisionChoiceQuestion):
        value["choice"] = names[probabilities.index(max(probabilities))]
    elif isinstance(question, DecisionScoreQuestion):
        value["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
    return DecisionAnswer(
        type=question.type,
        probabilities=dict(zip(names, probabilities)),
        label_mass=math.fsum(math.exp(logprob) for logprob in token_logprobs),
        **value,
    )
