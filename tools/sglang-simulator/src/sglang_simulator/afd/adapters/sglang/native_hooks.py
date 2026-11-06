"""Import native AFD modules, replacing only CPU-incompatible device boundaries.

Native classes keep their constructors, layer control and stage splitting.
No copied AFD loop is supplied. Model numerics and attention metadata are symbolic.
"""

import contextlib
import dataclasses
import importlib
import sys
import types
from pathlib import Path

from .loader import extract, temporary_modules


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


@dataclasses.dataclass
class ForwardContext:
    attn_backend: object = None
    moe_num_token_non_padded: object = None


@contextlib.contextmanager
def native_session(source_root, backend):
    """All imported source modules and patched contexts are restored on exit."""
    root = Path(source_root) / "python"
    context = [ForwardContext()]

    @contextlib.contextmanager
    def forward_context(value=None, forward_batch=None):
        previous = context[0]
        context[0] = value if value is not None else previous
        try:
            yield
        finally:
            context[0] = previous

    class ResidualStream:
        @classmethod
        def from_handoff(cls, hidden, residual, operation):
            return hidden, cls()

        def export(self, output):
            return output, None

    class MetadataGuard:
        @staticmethod
        def prepare_stage_metadata(**kwargs):
            pass

    replacements = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.afd",
        "sglang.srt.afd.model_adapters",
        "sglang.srt.batch_overlap",
        "sglang.srt.model_executor",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.layer_boundary",
        "sglang.srt.layers.layer_boundary.residual",
        "sglang.srt.utils",
    ):
        package = module(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        replacements[name] = package
    stubs = {
        "torch": backend.torch,
        "sglang.srt.runtime_context": module("sglang.srt.runtime_context"),
        "sglang.srt.layers.dp_attention": module(
            "sglang.srt.layers.dp_attention", set_dp_buffer_len=lambda *args: None
        ),
        "sglang.srt.utils.nvtx_utils": module(
            "sglang.srt.utils.nvtx_utils",
            operations_nvtx_range=lambda **kwargs: contextlib.nullcontext(),
        ),
        "sglang.srt.model_executor.forward_context": module(
            "sglang.srt.model_executor.forward_context",
            ForwardContext=ForwardContext,
            forward_context=forward_context,
            get_forward_context=lambda: context[0],
            has_forward_context=lambda: True,
        ),
        "sglang.srt.layers.moe.topk": module(
            "sglang.srt.layers.moe.topk",
            StandardTopKOutput=lambda weights, ids, x: types.SimpleNamespace(
                topk_weights=weights, topk_ids=ids
            ),
        ),
        "sglang.srt.layers.layer_boundary.residual.stream": module(
            "sglang.srt.layers.layer_boundary.residual.stream",
            ResidualStream=ResidualStream,
        ),
        "sglang.srt.layers.layer_boundary.residual.add_norm": module(
            "sglang.srt.layers.layer_boundary.residual.add_norm", PLAIN_ADD=object()
        ),
        "sglang.srt.afd.metadata": module(
            "sglang.srt.afd.metadata",
            FlashAttentionMetadataGuard=MetadataGuard,
            metadata_guard_for_backend=lambda name, default_guard: default_guard,
        ),
    }
    stubs["sglang.srt.runtime_context"].get_parallel = lambda: types.SimpleNamespace(
        attn_cp_size=1, attn_dcp_size=1, enable_prefill_cp=False
    )
    # These two pure metadata helpers are isolated from the device-heavy TBO
    # module; their bodies, including the split rule, come from native source.
    env = {"staticmethod": staticmethod}
    extract(
        root / "sglang/srt/batch_overlap/two_batch_overlap.py",
        env,
        ["_get_num_token_non_padded_cpu", "_split_num_token_non_padded"],
        "TboForwardBatchPreparer",
    )
    tbo = type(
        "TboForwardBatchPreparer",
        (),
        {
            name: env[name]
            for name in ("_get_num_token_non_padded_cpu", "_split_num_token_non_padded")
        },
    )
    stubs["sglang.srt.batch_overlap.two_batch_overlap"] = module(
        "sglang.srt.batch_overlap.two_batch_overlap", TboForwardBatchPreparer=tbo
    )
    replacements.update(stubs)
    with temporary_modules(replacements):
        saved = {
            name: value
            for name, value in sys.modules.items()
            if name.startswith("sglang.")
        }
        for name in list(saved):
            if name not in replacements:
                sys.modules.pop(name)
        try:
            loaded = {
                name: importlib.import_module("sglang.srt.afd." + name)
                for name in (
                    "contracts",
                    "config",
                    "pipeline",
                    "expert_exchange",
                    "model_adapters.qwen3_moe",
                )
            }
            yield types.SimpleNamespace(
                **{name.replace(".", "_"): value for name, value in loaded.items()}
            )
        finally:
            for name in list(sys.modules):
                if name.startswith("sglang."):
                    sys.modules.pop(name)
            sys.modules.update(saved)
