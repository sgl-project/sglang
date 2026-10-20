"""Shared fakes for the megakernel unit tests: one spec, one model, server arguments."""

from types import SimpleNamespace

import torch
from msgspec.structs import replace

from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.spec import (
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Step,
)
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, stage="base-a", runner_config="cpu", disabled="megakernel test fakes"
)

ARCH, ISA = "FakeMegaForCausalLM", "sm_103"


class FakeBuilt:
    """Covers steps of 1-4 rows and records the steps it ran."""

    draft_sampler = None

    def __init__(self, widths):
        self.widths = tuple(w for w in widths if w <= 4)
        self.steps, self.views = [], []

    def summary(self):
        return "fake"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        rows = int(input_ids.shape[0])
        return Step(mode="decode", rows=rows) if rows in self.widths else None

    def forward(self, step, original, input_ids, positions, forward_batch, **kw):
        self.steps.append(step.rows)
        self.views.append(step.view)
        return "megakernel"


def _build(model, hf_config, context):
    model.context = context
    return FakeBuilt(context.widths)


SPEC = MegakernelSpec(
    name="fake",
    requires=Requirements(isas=frozenset({ISA}), tp_sizes=frozenset({4})),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    max_rows=8,
    build=_build,
    api_version=registry.MEGAKERNEL_API_VERSION,
)
# an empty field accepts any value
ANY_TP = replace(SPEC, requires=replace(SPEC.requires, tp_sizes=frozenset()))
# a draft spec: builds no width, offers only the draft sampler
DRAFT = replace(
    SPEC,
    max_rows=0,
    build=lambda model, hf, ctx: SimpleNamespace(
        widths=(), summary=lambda: "draft sampler", draft_sampler=print
    ),
)


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "standard"


def batch(rows):
    return SimpleNamespace(batch_size=rows, input_ids=torch.zeros(rows))


def server_args(**kw):
    """Arguments as the resolution step sees them, with the graph config resolved."""
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "speculative_algorithm": "DSPARK",
        "speculative_num_draft_tokens": 6,
        "cuda_graph_config": default_cuda_graph_config(),
    }
    return ServerArgs(**{**defaults, **kw})
