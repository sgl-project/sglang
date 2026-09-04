"""Per-model config-time override declarations, one module per family,
mirroring the ``models/`` naming.

Importing this package is what registers them. An architecture may be claimed
by more than one module here -- one supplies its attention shape, another its
MoE runner -- but two of them must never declare the *same* field for it:
nobody would own that value, and which module supplied it would come down to
the order of the imports below. ``test_model_override_split.py`` forbids the
overlap, which is why this list needs no particular order.
"""

from sglang.srt.arg_groups.model_overrides import cohere2_moe  # noqa: F401
from sglang.srt.arg_groups.model_overrides import deepseek_v2  # noqa: F401
from sglang.srt.arg_groups.model_overrides import deepseek_v4  # noqa: F401
from sglang.srt.arg_groups.model_overrides import exaone  # noqa: F401
from sglang.srt.arg_groups.model_overrides import falcon_h1  # noqa: F401
from sglang.srt.arg_groups.model_overrides import gemma2_gemma3  # noqa: F401
from sglang.srt.arg_groups.model_overrides import gemma4  # noqa: F401
from sglang.srt.arg_groups.model_overrides import glm4_moe  # noqa: F401
from sglang.srt.arg_groups.model_overrides import gpt_oss  # noqa: F401
from sglang.srt.arg_groups.model_overrides import granitemoehybrid  # noqa: F401
from sglang.srt.arg_groups.model_overrides import inkling  # noqa: F401
from sglang.srt.arg_groups.model_overrides import interns2_mobius  # noqa: F401
from sglang.srt.arg_groups.model_overrides import kimi_k3  # noqa: F401
from sglang.srt.arg_groups.model_overrides import lfm2  # noqa: F401
from sglang.srt.arg_groups.model_overrides import llama4  # noqa: F401
from sglang.srt.arg_groups.model_overrides import mimo_v2  # noqa: F401
from sglang.srt.arg_groups.model_overrides import minicpm  # noqa: F401
from sglang.srt.arg_groups.model_overrides import minicpmv  # noqa: F401
from sglang.srt.arg_groups.model_overrides import minimax_m2  # noqa: F401
from sglang.srt.arg_groups.model_overrides import minimax_m3  # noqa: F401
from sglang.srt.arg_groups.model_overrides import moss_vl  # noqa: F401
from sglang.srt.arg_groups.model_overrides import muse_glimmer  # noqa: F401
from sglang.srt.arg_groups.model_overrides import nemotron_h  # noqa: F401
from sglang.srt.arg_groups.model_overrides import olmo2  # noqa: F401
from sglang.srt.arg_groups.model_overrides import qwen3_5  # noqa: F401
from sglang.srt.arg_groups.model_overrides import qwen3_moe  # noqa: F401
from sglang.srt.arg_groups.model_overrides import qwen3_vl  # noqa: F401
