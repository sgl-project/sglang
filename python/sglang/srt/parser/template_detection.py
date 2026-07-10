# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Template detection utilities for auto-detecting reasoning and tool-call parsers.

Provides rule-based detection of reasoning mode, reasoning parser, and tool-call
parser from chat templates and tokenizer vocabularies.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jinja2
import jinja2.ext
import jinja2.sandbox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemplateDetectionContext:
    template: str
    reasoning_config: Optional["ReasoningToggleConfig"]
    force_reasoning: bool
    vocab: set[str]

    def has_text(self, needle: str) -> bool:
        return needle in self.template

    def has_vocab(self, token: str) -> bool:
        return token in self.vocab

    def has_pattern(self, pattern: str, flags: int = 0) -> bool:
        return re.search(pattern, self.template, flags) is not None

    def has_vocab_pattern(self, pattern: str) -> bool:
        compiled = re.compile(pattern)
        return any(isinstance(tok, str) and compiled.search(tok) for tok in self.vocab)


@dataclass(frozen=True)
class DetectionRule:
    name: str
    value: object
    predicate: Callable[[TemplateDetectionContext], bool]


@dataclass(frozen=True)
class ReasoningToggleConfig:
    toggle_param: Optional[str] = None
    default_enabled: Optional[bool] = None
    special_case: Optional[str] = None
    effort_kwarg: Optional[str] = None

    @property
    def always_on(self) -> bool:
        return self.special_case == "always"


# ---------------------------------------------------------------------------
# Reasoning mode rules (detect toggle config from template)
# ---------------------------------------------------------------------------

REASONING_MODE_RULES = (
    DetectionRule(
        name="gpt_oss_channel_markers",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_text("<|channel|>"),
    ),
    DetectionRule(
        name="force_reasoning_pattern",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_pattern(r"<\|im_start\|>assistant\\n<think>\\n")
        and not ctx.has_text("enable_thinking")
        and not ctx.has_text("thinking"),
    ),
    DetectionRule(
        name="mistral_reasoning_effort",
        value=ReasoningToggleConfig(special_case="mistral"),
        predicate=lambda ctx: ctx.has_text("reasoning_effort")
        and ctx.has_text("[THINK]"),
    ),
    DetectionRule(
        name="explicit_enable_thinking_default_false",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=False
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        ),
    ),
    DetectionRule(
        name="nemotron_3_super_low_effort",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking",
            default_enabled=True,
            effort_kwarg="low_effort",
        ),
        predicate=lambda ctx: ctx.has_text("low_effort")
        and ctx.has_text("truncate_history_thinking"),
    ),
    DetectionRule(
        name="enable_thinking_default_true",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+enable_thinking\s*=\s*enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(
            r"enable_thinking\s+is\s+defined\s+and\s+(?:enable_thinking\s+is\s+false|not\s+enable_thinking)"
        )
        or ctx.has_pattern(
            r"enable_thinking\s+is\s+not\s+defined\s+or\s+enable_thinking"
        )
        or ctx.has_pattern(r"namespace\([^)]*enable_thinking\s*=\s*true"),
    ),
    DetectionRule(
        name="explicit_thinking_default_false",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        ),
    ),
    DetectionRule(
        name="thinking_default_true",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=True),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+thinking\s*=\s*thinking\s+if\s+thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(
            r"thinking\s+is\s+defined\s+and\s+(?:thinking\s+is\s+false|not\s+thinking)"
        )
        or ctx.has_pattern(r"thinking\s+is\s+not\s+defined\s+or\s+thinking")
        or ctx.has_pattern(r"namespace\([^)]*thinking\s*=\s*true"),
    ),
)


# ---------------------------------------------------------------------------
# Shared predicates for model-family detection
# ---------------------------------------------------------------------------


def _is_apertus2509(ctx):
    return ctx.has_vocab("<|inner_prefix|>")


def _is_gemma4(ctx):
    return ctx.has_text("<|channel>")


def _is_kimi(ctx):
    return ctx.has_text("◁think▷")


def _is_interns1(ctx):
    return ctx.has_text("default_thinking_sys") and ctx.reasoning_config == (
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
    )


def _is_mistral(ctx):
    return (
        ctx.reasoning_config is not None
        and ctx.reasoning_config.special_case == "mistral"
    )


def _is_gpt_oss(ctx):
    return ctx.has_text("<|channel|>")


def _is_kimi_k2(ctx):
    return ctx.has_vocab("<|tool_calls_section_begin|>")


def _is_nemotron_3(ctx):
    return ctx.has_text("truncate_history_thinking") and (
        ctx.reasoning_config is not None
        and ctx.reasoning_config.toggle_param == "enable_thinking"
        and ctx.reasoning_config.default_enabled is True
    )


def _is_glm45(ctx):
    return (
        (
            ctx.has_text("[gMASK]<sop>")
            or ctx.has_pattern(r"(?<!<)/nothink")
            or ctx.has_pattern(r"(?<!<)/think")
        )
        and ctx.has_vocab("<tool_call>")
        and ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
        and (ctx.has_vocab("<|user|>") or ctx.has_vocab("<|endoftext|>"))
    )


def _is_glm47(ctx):
    return _is_glm45(ctx) and ctx.has_pattern(
        r"\{\{[-\s]*['\"]<tool_call>['\"]\s*\+\s*tc\.name"
    )


def _is_xml_kv_tool_call(ctx):
    # Structural signature for the GLM-4.5 / GLM-4.6 style tool-call format
    # (`<tool_call>name<arg_key>k</arg_key>\n<arg_value>v</arg_value>...</tool_call>`).
    # Matches any model whose tokenizer carries `<arg_key>` and `<arg_value>` as
    # added tokens — e.g., inclusionAI/Ring-2.6, which borrows GLM's tool-call
    # format but doesn't share the `[gMASK]<sop>` / `enable_thinking` family
    # signature checked by `_is_glm45`.
    return ctx.has_vocab("<arg_key>") and ctx.has_vocab("<arg_value>")


def _is_deepseek_v31(ctx):
    return ctx.has_text("<｜tool▁calls▁begin｜>") and ctx.has_text("<｜tool▁sep｜>")


def _is_deepseek_v32(ctx):
    return ctx.has_text("<｜DSML｜function_calls>")


def _is_deepseek_v4(ctx):
    return ctx.has_text("<｜DSML｜tool_calls>")


def _is_hunyuan(ctx):
    # The shipping Hy3 tokenizer appends a shared suffix to each special token
    # (e.g. ``<tool_calls:opensource>``), so match the bare or suffixed form.
    tc = ctx.has_text("<tool_calls>") or ctx.has_vocab_pattern(
        r"^<tool_calls(?::[^>]+)?>$"
    )
    sep = ctx.has_text("<tool_sep>") or ctx.has_vocab_pattern(
        r"^<tool_sep(?::[^>]+)?>$"
    )
    return (tc and sep) or (
        ctx.has_text("reasoning_effort") and ctx.has_text("interleaved_thinking")
    )


def _is_poolside_v1(ctx):
    has_poolside_tool_format = (
        ctx.has_text("unescaped XML-like object")
        and ctx.has_text("<tool_call>function-name")
        and ctx.has_text("<arg_key>")
        and ctx.has_text("<arg_value>")
    )
    return has_poolside_tool_format or (
        ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=False)
        and not _is_hunyuan(ctx)
        and (ctx.has_text("<arg_key>") or ctx.has_vocab("<arg_key>"))
        and (ctx.has_text("<arg_value>") or ctx.has_vocab("<arg_value>"))
    )


def _is_mimo(ctx):
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=False
    )


def _is_minimax(ctx):
    return ctx.has_text("<minimax:tool_call>")


def _is_minicpm5(ctx):
    if ctx.has_vocab("<function") and ctx.has_vocab("<param"):
        return True
    return ctx.has_pattern(r"<function\s+name=") and ctx.has_pattern(r"<param\s+name=")


def _is_lfm2(ctx):
    return (
        ctx.has_text("<|tool_call_start|>") or ctx.has_vocab("<|tool_call_start|>")
    ) and (ctx.has_text("<|tool_call_end|>") or ctx.has_vocab("<|tool_call_end|>"))


def _is_step3p5(ctx):
    return ctx.has_pattern(r"Step-?3(?:\.|p)?[57]", re.IGNORECASE) or (
        ctx.has_text("reasoning_effort")
        and ctx.has_text("Reasoning: ")
        and _is_qwen3_coder(ctx)
    )


def _is_step3(ctx):
    return ctx.has_text("<steptml:invoke") or (
        ctx.has_text("<｜tool_calls_begin｜>") and ctx.has_text("<｜tool_sep｜>")
    )


def _is_qwen3_coder(ctx):
    return ctx.has_text("<function=") and ctx.has_text("<parameter=")


def _is_qwen3(ctx):
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=True
    )


def _is_deepseek_v3(ctx):
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="thinking", default_enabled=False
    )


def _is_deepseek_r1(ctx):
    return ctx.force_reasoning


def _is_deepseek_r1_think_tags(ctx):
    return not _is_lfm2(ctx) and (ctx.has_text("<think>") or ctx.has_text("</think>"))


# ---------------------------------------------------------------------------
# Reasoning parser rules
# ---------------------------------------------------------------------------

REASONING_PARSER_RULES = (
    DetectionRule(name="apertus2509", value="apertus2509", predicate=_is_apertus2509),
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="kimi", value="kimi", predicate=_is_kimi),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="nemotron_3", value="nemotron_3", predicate=_is_nemotron_3),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="hunyuan", value="hunyuan", predicate=_is_hunyuan),
    DetectionRule(name="poolside_v1", value="poolside_v1", predicate=_is_poolside_v1),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="minimax", value="minimax", predicate=_is_minimax),
    DetectionRule(name="step3p5", value="step3p5", predicate=_is_step3p5),
    DetectionRule(name="step3", value="step3", predicate=_is_step3),
    DetectionRule(name="qwen3", value="qwen3", predicate=_is_qwen3),
    DetectionRule(name="deepseek_v4", value="deepseek-v4", predicate=_is_deepseek_v4),
    DetectionRule(name="deepseek_v3", value="deepseek-v3", predicate=_is_deepseek_v3),
    DetectionRule(
        name="deepseek_r1_force", value="deepseek-r1", predicate=_is_deepseek_r1
    ),
    DetectionRule(
        name="deepseek_r1_think_tags",
        value="deepseek-r1",
        predicate=_is_deepseek_r1_think_tags,
    ),
)

# ---------------------------------------------------------------------------
# Tool-call parser rules (reuse shared predicates, different values)
# ---------------------------------------------------------------------------

TOOL_CALL_PARSER_RULES = (
    DetectionRule(name="apertus2509", value="apertus2509", predicate=_is_apertus2509),
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="minimax", value="minimax-m2", predicate=_is_minimax),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="deepseek_v4", value="deepseekv4", predicate=_is_deepseek_v4),
    DetectionRule(name="deepseek_v32", value="deepseekv32", predicate=_is_deepseek_v32),
    DetectionRule(name="deepseek_v31", value="deepseekv31", predicate=_is_deepseek_v31),
    DetectionRule(name="lfm2", value="lfm2", predicate=_is_lfm2),
    DetectionRule(name="glm47", value="glm47", predicate=_is_glm47),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="minicpm5", value="minicpm5", predicate=_is_minicpm5),
    DetectionRule(name="hunyuan", value="hunyuan", predicate=_is_hunyuan),
    DetectionRule(name="poolside_v1", value="poolside_v1", predicate=_is_poolside_v1),
    DetectionRule(name="step3p5", value="step3p5", predicate=_is_step3p5),
    DetectionRule(name="step3", value="step3", predicate=_is_step3),
    DetectionRule(
        name="xml_kv_tool_call", value="glm45", predicate=_is_xml_kv_tool_call
    ),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="qwen3_coder", value="qwen3_coder", predicate=_is_qwen3_coder),
    DetectionRule(name="qwen", value="qwen", predicate=_is_qwen3),
    DetectionRule(name="deepseek_v3", value="deepseekv3", predicate=_is_deepseek_v3),
    DetectionRule(name="deepseek_r1", value="deepseekv3", predicate=_is_deepseek_r1),
)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def build_detection_context(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[TemplateDetectionContext]:
    if template is None:
        return None
    vocab = set()
    if tokenizer is not None:
        try:
            vocab = set(tokenizer.get_vocab().keys())
        except Exception as e:
            logger.warning(
                "Failed to load tokenizer vocab for template detection: %s. "
                "Vocab-dependent detection rules will be skipped.",
                e,
            )
    return TemplateDetectionContext(
        template=template,
        reasoning_config=reasoning_config,
        force_reasoning=force_reasoning,
        vocab=vocab,
    )


def match_rules(
    ctx: TemplateDetectionContext,
    rules: Tuple[DetectionRule, ...],
    label: str,
) -> Optional[str]:
    for rule in rules:
        try:
            if rule.predicate(ctx):
                return rule.value
        except Exception as e:
            logger.warning(
                "Detection rule '%s' for %s raised an exception: %s. Skipping.",
                rule.name,
                label,
                e,
                exc_info=True,
            )
    return None


def detect_reasoning_pattern(
    template: Optional[str],
) -> Tuple[bool, Optional[ReasoningToggleConfig]]:
    """Detect if the chat template contains reasoning/thinking patterns."""
    if template is None:
        return False, None

    ctx = TemplateDetectionContext(
        template=template,
        reasoning_config=None,
        force_reasoning=False,
        vocab=set(),
    )
    for rule in REASONING_MODE_RULES:
        if rule.predicate(ctx):
            return rule.value.always_on, rule.value

    return False, None


def detect_reasoning_parser(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[str]:
    """Auto-detect which reasoning parser to use from the chat template."""
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        return None
    return match_rules(ctx, REASONING_PARSER_RULES, "reasoning parser")


def detect_tool_call_parser(
    template: Optional[str],
    tokenizer,
    reasoning_config: Optional[ReasoningToggleConfig] = None,
    force_reasoning: bool = False,
) -> Optional[str]:
    """Auto-detect which tool-call parser to use from the chat template."""
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        return None
    return match_rules(ctx, TOOL_CALL_PARSER_RULES, "tool-call parser")


def detect_inline_system_support(chat_template: Optional[str]) -> bool:
    """True if mid-conversation ``role: "system"`` renders inline; False if the
    template raises or silently drops it (then merge into the leading block).

    The probe requires the second system's sentinel to appear in the output —
    not raising isn't enough, since some templates ignore non-leading system."""
    if not chat_template:
        return False
    sentinel = "__sglang_inline_system_sentinel__"
    try:
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        rendered = env.from_string(chat_template).render(
            messages=[
                {"role": "system", "content": "t"},
                {"role": "user", "content": "t"},
                {"role": "system", "content": sentinel},
                {"role": "user", "content": "t"},
            ],
            add_generation_prompt=False,
        )
        return sentinel in rendered
    except jinja2.TemplateError:
        return False
    except Exception:
        return False


def _resolve_auto_parser(
    server_args,
    attr: str,
    ctx: TemplateDetectionContext,
    rules: Tuple[DetectionRule, ...],
    label: str,
) -> None:
    """Resolve a single auto parser, updating server_args in place."""
    detected = match_rules(ctx, rules, label)
    if detected:
        server_args.override(source="template-detection", **{attr: detected})
        logger.info(
            f"Auto-detected --{attr.replace('_', '-')} as '{detected}' from chat template"
        )
    else:
        logger.warning(
            f"--{attr.replace('_', '-')}=auto specified but could not detect "
            f"{label} from chat template. Disabling {label}."
        )
        server_args.override(source="template-detection", **{attr: None})


def _load_explicit_jinja_template(chat_template_arg: Optional[str]) -> Optional[str]:
    if not chat_template_arg or not isinstance(chat_template_arg, str):
        return None
    if not chat_template_arg.endswith(".jinja") or not os.path.exists(
        chat_template_arg
    ):
        return None
    with open(chat_template_arg, encoding="utf-8") as f:
        return f.read().replace("\\n", "\n")


def _disable_auto_parser(server_args, attr: str, label: str) -> None:
    logger.warning(
        f"--{attr.replace('_', '-')}=auto specified but could not detect "
        f"{label} from chat template. Disabling {label}."
    )
    server_args.override(source="template-detection", **{attr: None})


def _resolve_architecture_auto_parsers(server_args) -> None:
    from sglang.srt.utils.hf_transformers_utils import get_config

    config = get_config(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=getattr(server_args, "revision", None),
        model_config_parser=getattr(server_args, "model_config_parser", "auto"),
    )
    architectures = getattr(config, "architectures", None) or []
    arch = architectures[0] if architectures else ""

    if "DeepseekV4" in arch:
        reasoning_parser, tool_call_parser = "deepseek-v4", "deepseekv4"
    elif "DeepseekV3" in arch:
        reasoning_parser, tool_call_parser = "deepseek-v3", "deepseekv32"
    else:
        return

    for attr, detected in (
        ("reasoning_parser", reasoning_parser),
        ("tool_call_parser", tool_call_parser),
    ):
        if getattr(server_args, attr) == "auto":
            server_args.override(source="template-detection", **{attr: detected})
            logger.info(
                f"Auto-detected --{attr.replace('_', '-')} as '{detected}' "
                f"from model architecture '{arch}'"
            )


def resolve_auto_parsers(server_args) -> None:
    """Resolve --reasoning-parser=auto and --tool-call-parser=auto before scheduler.

    This performs a lightweight tokenizer load to detect parsers from the chat
    template. Called early in engine init before scheduler subprocesses are spawned.
    """
    needs_reasoning = server_args.reasoning_parser == "auto"
    needs_tool_call = server_args.tool_call_parser == "auto"

    if not needs_reasoning and not needs_tool_call:
        return

    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    chat_template_arg = getattr(server_args, "chat_template", None)
    try:
        explicit_jinja_template = _load_explicit_jinja_template(chat_template_arg)
    except Exception as e:
        logger.warning("Failed to load explicit Jinja chat template: %s", e)
        explicit_jinja_template = None
    has_explicit_template_without_detection = (
        chat_template_arg is not None and explicit_jinja_template is None
    )

    tokenizer = None
    try:
        tokenizer = get_tokenizer(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for auto-detection: {e}")

    template = explicit_jinja_template
    if template is None and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)

    force_reasoning, reasoning_config = detect_reasoning_pattern(template)
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
        if has_explicit_template_without_detection:
            logger.warning(
                "--chat-template=%s is explicit but is not a readable Jinja template, so "
                "parser auto-detection from chat template is not available.",
                chat_template_arg,
            )
        else:
            try:
                _resolve_architecture_auto_parsers(server_args)
            except Exception as e:
                logger.warning(
                    "Failed to load model config for architecture-based auto-detection: %s",
                    e,
                )
        if needs_reasoning:
            if server_args.reasoning_parser == "auto":
                _disable_auto_parser(
                    server_args, "reasoning_parser", "reasoning parser"
                )
        if needs_tool_call:
            if server_args.tool_call_parser == "auto":
                _disable_auto_parser(
                    server_args, "tool_call_parser", "tool-call parser"
                )
        return

    if needs_reasoning:
        _resolve_auto_parser(
            server_args,
            "reasoning_parser",
            ctx,
            REASONING_PARSER_RULES,
            "reasoning parser",
        )

    if needs_tool_call:
        _resolve_auto_parser(
            server_args,
            "tool_call_parser",
            ctx,
            TOOL_CALL_PARSER_RULES,
            "tool-call parser",
        )
