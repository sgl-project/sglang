# Copyright 2024 SGLang Team (internal fork)
# Early exit Qwen3 for sequence classification / embedding extraction.
# Reuses EarlyExitMixin from qwen2_early_exit.py.

from sglang.srt.models.qwen2_early_exit import EarlyExitMixin
from sglang.srt.models.qwen3 import Qwen3ForCausalLM


class Qwen3ForEarlyExitCausalLM(EarlyExitMixin, Qwen3ForCausalLM):
    pass


EntryClass = Qwen3ForEarlyExitCausalLM
