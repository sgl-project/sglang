"""Streamed tool calls must not depend on how the output is chunked.

Every registered tool-call parser receives the same two-call model output,
streamed three ways: one call per chunk, both calls in one chunk, and one token
per chunk. The client must receive both calls, each name once and the full
arguments, with no tool-call markup in the content.
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import re
import unittest
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    Tool,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get the current weather in a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        ),
    ),
    Tool(
        type="function",
        function=Function(
            name="get_time",
            description="Get the current time in a timezone",
            parameters={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        ),
    ),
]

EXPECTED_CALLS = [
    ("get_weather", {"city": "Paris", "unit": "celsius"}),
    ("get_time", {"timezone": "UTC"}),
]

MODES = ("one_call_per_chunk", "both_calls_in_one_chunk", "one_token_per_chunk")


@dataclass(frozen=True)
class Sample:
    """The model output for EXPECTED_CALLS, in one parser's format."""

    calls: Tuple[str, str]
    prefix: str = ""
    separator: str = ""
    suffix: str = ""
    # Markers that are a single token in the model's vocabulary.
    special_tokens: Tuple[str, ...] = ()

    def text(self) -> str:
        return self.prefix + self.separator.join(self.calls) + self.suffix

    def one_call_per_chunk(self) -> List[str]:
        first, second = self.calls
        return [self.prefix + first, self.separator + second + self.suffix]

    def both_calls_in_one_chunk(self) -> List[str]:
        return [self.text()]

    def one_token_per_chunk(self) -> List[str]:
        # Text between special tokens is split per character, which covers
        # every way a tokenizer can split it.
        if not self.special_tokens:
            return list(self.text())
        tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(map(re.escape, tokens)) + ")"
        chunks = []
        for part in re.split(pattern, self.text()):
            chunks.extend([part] if part in self.special_tokens else list(part))
        return chunks


def _calls(call: str, arg: str = "", arg_sep: str = "") -> Tuple[str, str]:
    """EXPECTED_CALLS rendered with the `call` template.

    `call` can use {i}, {name}, {args_json}, and {args}: each argument rendered
    with the `arg` template ({k}, {v}), joined by `arg_sep`.
    """
    return tuple(
        call.format(
            i=i,
            name=name,
            args_json=json.dumps(args),
            args=arg_sep.join(arg.format(k=k, v=v) for k, v in args.items()),
        )
        for i, (name, args) in enumerate(EXPECTED_CALLS)
    )


def _json_calls(call: str, args_key: str = "arguments") -> Tuple[str, str]:
    """EXPECTED_CALLS as `{"name": ..., args_key: ...}` objects, at {json} in `call`."""
    return tuple(
        call.format(json=json.dumps({"name": name, args_key: args}))
        for name, args in EXPECTED_CALLS
    )


# Each sample follows the model's chat template for assistant tool calls, or
# the detector's documented format where the template is not public.

_HERMES = Sample(
    calls=_json_calls("<tool_call>\n{json}\n</tool_call>"),
    separator="\n",
    special_tokens=("<tool_call>", "</tool_call>"),
)

_DSV3_TOKENS = (
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁sep｜>",
)

_QWEN3_CODER = Sample(
    calls=_calls(
        "<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>",
        "<parameter={k}>\n{v}\n</parameter>\n",
    ),
    separator="\n",
    special_tokens=("<tool_call>", "</tool_call>"),
)

_ARG_TAGS = (
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
)

# zai-org/GLM-4.7; RecursiveSI/flint renders the same call bodies.
_GLM47 = Sample(
    calls=_calls(
        "<tool_call>{name}{args}</tool_call>",
        "<arg_key>{k}</arg_key><arg_value>{v}</arg_value>",
    ),
    special_tokens=_ARG_TAGS,
)


def _dsml(block: str, invoke: str = "invoke", parameter: str = "parameter"):
    # Only the DSML marker itself is a single token.
    return Sample(
        calls=tuple(
            f'<｜DSML｜{invoke} name="{name}">\n'
            + "".join(
                f'<｜DSML｜{parameter} name="{k}" string="true">'
                f"{v}</｜DSML｜{parameter}>\n"
                for k, v in args.items()
            )
            + f"</｜DSML｜{invoke}>"
            for name, args in EXPECTED_CALLS
        ),
        prefix=f"\n\n<｜DSML｜{block}>\n",
        separator="\n",
        suffix=f"\n</｜DSML｜{block}>",
        special_tokens=("｜DSML｜",),
    )


# thinkingmachines/Inkling-Small; inkling2 parses the same format.
_INKLING = Sample(
    calls=tuple(
        f"<|message_model|>{name}<|content_invoke_tool_json|>"
        + json.dumps({"name": name, "args": args}, separators=(",", ":"))
        + "<|end_message|>"
        for name, args in EXPECTED_CALLS
    ),
    special_tokens=(
        "<|message_model|>",
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
    ),
)

SAMPLES = {
    # swiss-ai/Apertus-8B-Instruct-2509
    "apertus2509": Sample(
        calls=tuple(json.dumps({name: args}) for name, args in EXPECTED_CALLS),
        prefix="<|tools_prefix|>[",
        separator=", ",
        suffix="]<|tools_suffix|>",
        special_tokens=("<|tools_prefix|>", "<|tools_suffix|>"),
    ),
    # reflection/Beam-1.0-501B-A23B, after the reasoning parser consumed the
    # first `<|start|>assistant`.
    "beam": Sample(
        calls=_calls(
            " to={name}<|channel|>analysis <|constrain|>xml<|message|>{args}",
            '<rfl:parameter name="{k}">\n{v}\n</rfl:parameter>',
            arg_sep="\n",
        ),
        separator="<|end|><|start|>assistant",
        suffix="<|call|>",
        special_tokens=(
            "<|start|>",
            "<|end|>",
            "<|channel|>",
            "<|constrain|>",
            "<|message|>",
            "<|call|>",
            '<rfl:parameter name="',
            "\n</rfl:parameter>",
        ),
    ),
    # CohereLabs/command-a-plus-05-2026
    "cohere_command4": Sample(
        calls=tuple(
            json.dumps({"tool_call_id": str(i), "tool_name": name, "parameters": args})
            for i, (name, args) in enumerate(EXPECTED_CALLS)
        ),
        prefix="<|START_ACTION|>[\n    ",
        separator=",\n    ",
        suffix="\n]<|END_ACTION|>",
        special_tokens=("<|START_ACTION|>", "<|END_ACTION|>"),
    ),
    "deepseekv3": Sample(
        calls=_calls(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n"
            "```json\n{args_json}\n```<｜tool▁call▁end｜>"
        ),
        prefix="<｜tool▁calls▁begin｜>",
        separator="\n",
        suffix="<｜tool▁calls▁end｜>",
        special_tokens=_DSV3_TOKENS,
    ),
    "deepseekv31": Sample(
        calls=_calls(
            "<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{args_json}<｜tool▁call▁end｜>"
        ),
        prefix="<｜tool▁calls▁begin｜>",
        suffix="<｜tool▁calls▁end｜>",
        special_tokens=_DSV3_TOKENS,
    ),
    "deepseekv32": _dsml("function_calls"),
    "deepseekv4": _dsml("tool_calls"),
    # deepseek-ai/DeepSeek-V4.1
    "deepseekv41": _dsml(" calls", " invoke", " parameter"),
    # dots-studio/dots3-note-prev
    "dots": Sample(
        calls=_calls(
            '<dots_function_call>\n<invoke name="{name}">{args}\n</invoke>\n'
            "</dots_function_call>",
            '\n<parameter name="{k}">\n{v}\n</parameter>',
        ),
        prefix="\n",
        separator="\n",
    ),
    "ernie55": Sample(
        calls=_calls(
            '<invoke name="{name}">\n{args}</invoke>\n',
            '<parameter name="{k}">{v}</parameter>\n',
        ),
        prefix="<ernie:tool_call>\n",
        suffix="</ernie:tool_call>",
    ),
    # ai-sage/GigaChat3.5-432B-A28B
    "gigachat35": Sample(
        calls=_calls(
            '<｜GCML｜invoke name="{name}">\n{args}</｜GCML｜invoke>',
            '<｜GCML｜parameter name="{k}" string="true">{v}</｜GCML｜parameter>\n',
        ),
        prefix="<｜GCML｜tool_calls>\n",
        separator="\n",
        suffix="\n</｜GCML｜tool_calls>",
        special_tokens=("｜GCML｜",),
    ),
    # google/gemma-4-31B-it
    "gemma4": Sample(
        calls=tuple(
            f"<|tool_call>call:{name}"
            + "{"
            + ",".join(f'{k}:<|"|>{v}<|"|>' for k, v in args.items())
            + "}<tool_call|>"
            for name, args in EXPECTED_CALLS
        ),
        special_tokens=("<|tool_call>", "<tool_call|>", '<|"|>'),
    ),
    "glm": Sample(
        calls=_calls(
            "<tool_call>{name}\n{args}</tool_call>",
            "<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n",
        ),
        separator="\n",
        special_tokens=_ARG_TAGS,
    ),
    "glm47": _GLM47,
    "flint": _GLM47,
    # openai/gpt-oss-20b, as generated: the recipient is in the channel header.
    "gpt-oss": Sample(
        calls=_calls(
            "<|start|>assistant<|channel|>commentary to=functions.{name} "
            "<|constrain|>json<|message|>{args_json}<|call|>"
        ),
        special_tokens=(
            "<|start|>",
            "<|channel|>",
            "<|constrain|>",
            "<|message|>",
            "<|call|>",
        ),
    ),
    # ibm-granite/granite-4.2-3b
    "granite_toolcall_parser": Sample(
        calls=_QWEN3_CODER.calls,
        separator="\n",
        suffix="\n",
        special_tokens=("<tool_call>", "</tool_call>"),
    ),
    "hermes": _HERMES,
    # tencent/Hy3-preview
    "hunyuan": Sample(
        calls=_calls(
            "<tool_call>{name}<tool_sep>\n{args}</tool_call>",
            "<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n",
        ),
        prefix="<tool_calls>\n",
        separator="\n",
        suffix="\n</tool_calls>",
        special_tokens=_ARG_TAGS + ("<tool_calls>", "</tool_calls>", "<tool_sep>"),
    ),
    "inkling": _INKLING,
    "inkling2": _INKLING,
    # Detokenized with spaces_between_special_tokens, the server default.
    "interns1": Sample(
        calls=_json_calls(
            "<|action_start|> <|plugin|>\n{json}<|action_end|>", args_key="parameters"
        ),
        separator="\n",
        special_tokens=("<|action_start|>", "<|plugin|>", "<|action_end|>"),
    ),
    # IQuestLab/IQuest-Q1
    "iquest_q1": Sample(
        calls=_calls(
            "<iquest_tool_call>{name}{args}</iquest_tool_call>",
            "<arg_key>{k}</arg_key><arg_value>{v}</arg_value>",
        ),
        special_tokens=("<iquest_tool_call>", "</iquest_tool_call>") + _ARG_TAGS[2:],
    ),
    # IFM/K2-Horizon-0.9B
    "k2_horizon": Sample(
        calls=_calls(
            "<ifm|tool_call>{name}\n{args}</ifm|tool_call>",
            "<ifm|arg_key>{k}</ifm|arg_key>\n<ifm|arg_value>{v}</ifm|arg_value>\n",
        ),
        prefix="<ifm|tool_calls>\n",
        separator="\n",
        suffix="\n</ifm|tool_calls>",
        special_tokens=(
            "<ifm|tool_calls>",
            "</ifm|tool_calls>",
            "<ifm|tool_call>",
            "</ifm|tool_call>",
            "<ifm|arg_key>",
            "</ifm|arg_key>",
            "<ifm|arg_value>",
            "</ifm|arg_value>",
        ),
    ),
    "kimi_k2": Sample(
        calls=_calls(
            "<|tool_call_begin|>functions.{name}:{i}<|tool_call_argument_begin|>"
            "{args_json}<|tool_call_end|>"
        ),
        prefix="<|tool_calls_section_begin|>",
        suffix="<|tool_calls_section_end|>",
        special_tokens=(
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
        ),
    ),
    "kimi_k3": Sample(
        calls=tuple(
            f'<|open|>call tool="{name}" index="{i}"<|sep|>'
            + "".join(
                f'<|open|>argument key="{k}" type="string"<|sep|>'
                f"{v}<|close|>argument<|sep|>"
                for k, v in args.items()
            )
            + "<|close|>call<|sep|>"
            for i, (name, args) in enumerate(EXPECTED_CALLS, 1)
        ),
        prefix="<|open|>tools<|sep|>",
        suffix="<|close|>tools<|sep|>",
        special_tokens=("<|open|>", "<|close|>", "<|sep|>"),
    ),
    # LiquidAI/LFM2.5-1.2B-Instruct
    "lfm2": Sample(
        calls=_calls("{name}({args})", "{k}={v!r}", arg_sep=", "),
        prefix="<|tool_call_start|>[",
        separator=", ",
        suffix="]<|tool_call_end|>",
        special_tokens=("<|tool_call_start|>", "<|tool_call_end|>"),
    ),
    # inclusionAI/Ling-3.0-flash
    "ling3": Sample(
        calls=_calls(
            "<tool_call>{name}\n{args}\n</tool_call>",
            "<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>",
        ),
        separator="\n",
        special_tokens=_ARG_TAGS,
    ),
    "llama3": Sample(
        calls=_json_calls("{json}", args_key="parameters"),
        prefix="<|python_tag|>",
        separator=";",
        special_tokens=("<|python_tag|>",),
    ),
    # XiaomiMiMo/MiMo-V2-Flash
    "mimo": Sample(
        calls=_calls(
            "<tool_call>\n<function={name}>\n{args}</function>\n</tool_call>",
            "<parameter={k}>{v}</parameter>\n",
        ),
        special_tokens=("<tool_call>", "</tool_call>"),
    ),
    # openbmb/MiniCPM5-2B
    "minicpm5": Sample(
        calls=_calls(
            '<function name="{name}">{args}</function>',
            '<param name="{k}">{v}</param>',
        ),
        separator="\n",
        special_tokens=("<function", "</function>", "<param", "</param>"),
    ),
    # MiniMaxAI/MiniMax-M2
    "minimax-m2": Sample(
        calls=_calls(
            '<invoke name="{name}">\n{args}</invoke>',
            '<parameter name="{k}">{v}</parameter>\n',
        ),
        prefix="\n<minimax:tool_call>\n",
        separator="\n",
        suffix="\n</minimax:tool_call>",
        special_tokens=("<minimax:tool_call>", "</minimax:tool_call>"),
    ),
    # MiniMaxAI/MiniMax-M3
    "minimax-m3": Sample(
        calls=_calls(
            ']<]minimax[>[<invoke name="{name}">{args}]<]minimax[>[</invoke>',
            "]<]minimax[>[<{k}>{v}]<]minimax[>[</{k}>",
        ),
        prefix="]<]minimax[>[<tool_call>\n",
        separator="\n",
        suffix="\n]<]minimax[>[</tool_call>",
        special_tokens=("]<]minimax[>[", "<tool_call>", "</tool_call>"),
    ),
    "mistral": Sample(
        calls=_json_calls("{json}"),
        prefix="[TOOL_CALLS] [",
        separator=", ",
        suffix="]",
        special_tokens=("[TOOL_CALLS]",),
    ),
    # meta-models/Muse-Glimmer-30B, after the prompt's `<|start|>assistant`.
    "muse": Sample(
        calls=_calls(
            " to={name}<|message|><atem:function_calls>\n"
            '<atem:invoke name="{name}">\n{args}</atem:invoke>\n</atem:function_calls>',
            '<atem:parameter name="{k}">{v}</atem:parameter>\n',
        ),
        separator="<|eom|><|start|>assistant",
        suffix="<|eot|>",
        special_tokens=("<|message|>", "<|eom|>", "<|start|>", "<|eot|>"),
    ),
    "nanbeige": _QWEN3_CODER,
    # poolside/Laguna-XS.2
    "poolside_v1": Sample(
        calls=_calls(
            "<tool_call>{name}\n{args}</tool_call>\n",
            "<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n",
        ),
        special_tokens=("<tool_call>", "</tool_call>"),
    ),
    "pythonic": Sample(
        calls=_calls("{name}({args})", '{k}="{v}"', arg_sep=", "),
        prefix="[",
        separator=", ",
        suffix="]",
    ),
    "qwen": _HERMES,
    "qwen3_coder": _QWEN3_CODER,
    "spark25": Sample(calls=_GLM47.calls),
    # stepfun-ai/step3
    "step3": Sample(
        calls=_calls(
            "<｜tool_call_begin｜>\nfunction<｜tool_sep｜>"
            '<steptml:invoke name="{name}">{args}</steptml:invoke><｜tool_call_end｜>',
            '<steptml:parameter name="{k}">{v}</steptml:parameter>',
        ),
        prefix="<｜tool_calls_begin｜>\n",
        separator="\n",
        suffix="\n<｜tool_calls_end｜>",
    ),
    "step3p5": _QWEN3_CODER,
    # arcee-ai/Trinity-Mini
    "trinity": Sample(calls=_HERMES.calls, separator="\n"),
    # XingChen-AGI/Xing4.0-29B-A4B
    "xing4_0": Sample(
        calls=_calls(
            "<tool_call>{name}{args}</tool_call>",
            "<param_key>{k}</param_key><param_value>{v}</param_value>",
        ),
        special_tokens=("<tool_call>", "</tool_call>"),
    ),
}

# GigaChat4 inherits the GCML tool-call format from GigaChat3.5.
SAMPLES["gigachat4"] = SAMPLES["gigachat35"]

# Parsers that cannot be driven from a fixed sample, with the reason.
NOT_COVERED = {
    "gigachat3": "the chat template renders only one tool call per message",
    "response_template": (
        "built from a per-model response template; chunking is covered in "
        "test/registered/unit/parser/test_chat_parsing.py"
    ),
}

_LATER_CALLS_TO_CONTENT = "calls after the first go to content"
_SECTION_END_DROPS_CALLS = "section end discards buffered calls"


# (parser, mode) pairs that fail on main. Remove an entry when its fix lands.
KNOWN_FAILURES = {
    ("hermes", "both_calls_in_one_chunk"): "`</tool_call>` leaks into content",
    ("mistral", "one_call_per_chunk"): _LATER_CALLS_TO_CONTENT,
    ("mistral", "both_calls_in_one_chunk"): _LATER_CALLS_TO_CONTENT,
    ("mistral", "one_token_per_chunk"): _LATER_CALLS_TO_CONTENT,
    ("step3", "one_call_per_chunk"): _SECTION_END_DROPS_CALLS,
    ("step3", "both_calls_in_one_chunk"): _SECTION_END_DROPS_CALLS,
}

_REQUEST = ChatCompletionRequest(
    model="test",
    messages=[{"role": "user", "content": "Weather in Paris and the time in UTC?"}],
    tools=TOOLS,
)


def _stream(parser_name: str, chunks: List[str]):
    """Feed chunks through the parser as the chat server does for a stream.

    Returns the streamed content and, per tool index, the names sent and the
    concatenated arguments.
    """
    parser = FunctionCallParser(tools=TOOLS, tool_call_parser=parser_name)
    results = [parser.parse_stream_chunk(chunk) for chunk in chunks]
    results.append(parser.parse_stream_end())

    content = "".join(text for text, _ in results)
    names, arguments = defaultdict(list), defaultdict(str)
    for _, calls in results:
        for call in calls:
            if call.name:
                names[call.tool_index].append(call.name)
            arguments[call.tool_index] += call.parameters

    # The server sends the arguments of the last call that were not streamed.
    flush = OpenAIServingChat._check_for_unstreamed_tool_args(
        None,
        parser,
        content={"meta_info": {"id": "chatcmpl-test"}},
        request=_REQUEST,
        index=0,
        created=0,
    )
    if flush:
        delta = json.loads(flush[len("data: ") :])["choices"][0]["delta"]
        for call in delta["tool_calls"]:
            arguments[call["index"]] += call["function"]["arguments"]
    return content, names, arguments


class TestStreamingChunkBoundaries(CustomTestCase):
    def test_every_registered_parser_is_covered(self):
        self.assertEqual(
            set(SAMPLES) | set(NOT_COVERED), set(FunctionCallParser.ToolCallParserEnum)
        )

    def test_samples_parse_without_streaming(self):
        for name, sample in SAMPLES.items():
            with self.subTest(parser=name):
                parser = FunctionCallParser(tools=TOOLS, tool_call_parser=name)
                content, calls = parser.parse_non_stream(sample.text())
                self.assertEqual(content.strip(), "")
                self.assertEqual(
                    [(c.name, json.loads(c.parameters)) for c in calls],
                    EXPECTED_CALLS,
                )

    def _check_stream(self, parser_name: str, chunks: List[str]):
        content, names, arguments = _stream(parser_name, chunks)
        self.assertEqual(content.strip(), "", "tool-call text leaked into content")
        self.assertEqual(sorted(arguments), list(range(len(EXPECTED_CALLS))))
        received = [
            (names.get(i), json.loads(arguments[i] or "{}")) for i in sorted(arguments)
        ]
        self.assertEqual(received, [([name], args) for name, args in EXPECTED_CALLS])


def _make_test(parser_name: str, mode: str):
    def test(self):
        self._check_stream(parser_name, getattr(SAMPLES[parser_name], mode)())

    if (parser_name, mode) in KNOWN_FAILURES:
        test = unittest.expectedFailure(test)
    return test


for _name in SAMPLES:
    for _mode in MODES:
        _test_name = "test_{}_{}".format(re.sub(r"\W", "_", _name), _mode)
        setattr(TestStreamingChunkBoundaries, _test_name, _make_test(_name, _mode))


if __name__ == "__main__":
    unittest.main()
