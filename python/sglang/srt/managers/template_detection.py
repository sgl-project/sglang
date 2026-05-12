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
import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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
    return ctx.has_text("truncate_history_thinking") and ctx.reasoning_config == (
        ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
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


def _is_mimo(ctx):
    return ctx.reasoning_config == ReasoningToggleConfig(
        toggle_param="enable_thinking", default_enabled=False
    )


def _is_minimax(ctx):
    return ctx.has_text("<minimax:tool_call>")


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
    return ctx.has_text("<think>") or ctx.has_text("</think>")


# ---------------------------------------------------------------------------
# Reasoning parser rules
# ---------------------------------------------------------------------------

REASONING_PARSER_RULES = (
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="kimi", value="kimi", predicate=_is_kimi),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="nemotron_3", value="nemotron_3", predicate=_is_nemotron_3),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
    DetectionRule(name="minimax", value="minimax", predicate=_is_minimax),
    DetectionRule(name="qwen3", value="qwen3", predicate=_is_qwen3),
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
    DetectionRule(name="gemma4", value="gemma4", predicate=_is_gemma4),
    DetectionRule(name="gpt_oss", value="gpt-oss", predicate=_is_gpt_oss),
    DetectionRule(name="kimi_k2", value="kimi_k2", predicate=_is_kimi_k2),
    DetectionRule(name="minimax", value="minimax-m2", predicate=_is_minimax),
    DetectionRule(name="interns1", value="interns1", predicate=_is_interns1),
    DetectionRule(name="mistral", value="mistral", predicate=_is_mistral),
    DetectionRule(name="glm45", value="glm45", predicate=_is_glm45),
    DetectionRule(name="mimo", value="mimo", predicate=_is_mimo),
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
                logger.info(
                    "Detected %s '%s' from template rule '%s'.",
                    label,
                    rule.value,
                    rule.name,
                )
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
            logger.info(
                "Detected reasoning config '%s' from template rule '%s'.",
                rule.value,
                rule.name,
            )
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
        setattr(server_args, attr, detected)
        logger.info(
            f"Auto-detected --{attr.replace('_', '-')} as '{detected}' from chat template"
        )
    else:
        logger.warning(
            f"--{attr.replace('_', '-')}=auto specified but could not detect "
            f"{label} from chat template. Disabling {label}."
        )
        setattr(server_args, attr, None)


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

    try:
        tokenizer = get_tokenizer(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        template = getattr(tokenizer, "chat_template", None)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for auto-detection: {e}")
        if needs_reasoning:
            logger.warning(
                "--reasoning-parser=auto specified but could not detect "
                "reasoning parser from chat template. Disabling reasoning parser."
            )
            server_args.reasoning_parser = None
        if needs_tool_call:
            logger.warning(
                "--tool-call-parser=auto specified but could not detect "
                "tool-call parser from chat template. Disabling tool-call parser."
            )
            server_args.tool_call_parser = None
        return

    force_reasoning, reasoning_config = detect_reasoning_pattern(template)
    ctx = build_detection_context(
        template, tokenizer, reasoning_config, force_reasoning
    )
    if ctx is None:
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
