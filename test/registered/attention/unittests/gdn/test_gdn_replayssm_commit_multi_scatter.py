"""The verify commit's scatters as one multi-pair launch each (SGLANG_OPT_SPEC_COMMIT_MULTI_SCATTER): the GDN ReplaySSM
commit's conv-window rollback and the PLE side-state rollback, bit for bit the stock per-pair launches.

The state lives in one arena laid out by the state entry layout with Qwen3.8's PLE declarations (strided bf16 short-conv
and int64 n-gram views beside the conv and checkpoint parts). Each arm commits its own copy of every tensor, and the whole
arena, the scratch, the rings and the cursors are compared as bytes, padding and untouched slots included.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="the multi-pair scatters are CUDA Triton kernels",
    ),
]

DEVICE = "cuda"
LAYERS, H, HV, D, RING = 2, 2, 8, 128, 16
CONV_DIM, CONV_KM1 = (2 * H + HV) * D, 3
WIDTH = 4  # draft tokens per verify
SLOTS, ROWS = 16, 6  # state slots; request rows of the verify scratch
EOS = 7
# A padded row without a slot (-1), a rejected row (step -1) and a step past the verify window.
SLOT_ROWS = [3, 7, -1, 1, 9, 2]
STEP_ROWS = [2, 0, 1, -1, WIDTH, 3]
# Track slots are their own allocations, disjoint from the working slots.
TRACK_ROWS = [10, -1, 12, 4, 13, 5]
TRACK_STEP_ROWS = [1, -1, 0, 3, -1, 2]
SRC_ROWS = [4, 0, 5, 1, 3, 2]  # PP stable rows: the verify scratch row of each request


def _modules():
    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as gdn
    from sglang.kernels.ops.mamba import mamba_state_scatter_triton as scatter
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as hla

    return gdn, scatter, hla


def _layout(conv_dtype):
    """Qwen3.8's state declarations, as the KV cache configurator makes them (a narrower hidden size)."""
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
    from sglang.srt.mem_cache.kv.decl import StateGeometry, with_state_ple
    from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec

    config = Qwen4ExpTextConfig(hidden_size=256, ple_layer_ids=[3, 6], eos_token_id=EOS)
    geometry = with_state_ple(
        StateGeometry(
            conv_state_shapes=((CONV_DIM, CONV_KM1),),
            conv_dtype=conv_dtype,
            temporal_state_shape=(HV, D, D),
            temporal_dtype=torch.bfloat16,
        ),
        short_conv_layer_ids=tuple(config.short_conv_layer_ids),
        short_conv_state_shape=config.short_conv_state_shape,
        ngram_context_len=config.ngram_context_len,
        ngram_eos_token_id=EOS,
        speculative_num_draft_tokens=WIDTH,
    )
    spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=LAYERS,
        grow_direction="up",
        conv_state_shapes=geometry.conv_state_shapes,
        conv_dtype=conv_dtype,
        temporal_state_shape=geometry.temporal_state_shape,
        temporal_dtype=geometry.temporal_dtype,
        extra_parts=geometry.extra_parts,
    )
    return geometry, spec


def _views(spec, raw):
    return spec.entry_layout().views(raw, SLOTS, entry_bytes=spec.entry_bytes())


def _window(buf, dedup: bool):
    if not dedup:
        return buf
    # MambaPool's deduplicated sliding window: step t's window is buf[..., t : t + KM1].
    return buf.as_strided(
        (LAYERS, ROWS, WIDTH, CONV_DIM, CONV_KM1),
        (buf.stride(0), buf.stride(1), 1, buf.stride(2), 1),
    )


def _case(seed: int, *, conv_dtype=torch.bfloat16, dedup: bool = False):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    geometry, spec = _layout(conv_dtype)

    def rand(shape, dtype, scale=0.1):
        if dtype == torch.int64:
            return torch.randint(0, 1 << 40, shape, device=DEVICE, generator=gen)
        return (torch.randn(shape, device=DEVICE, generator=gen) * scale).to(dtype)

    # Random bytes everywhere, then finite values in the parts the commit reads or writes.
    raw = torch.randint(
        0,
        256,
        (SLOTS * spec.entry_bytes(),),
        dtype=torch.uint8,
        device=DEVICE,
        generator=gen,
    )
    for view in _views(spec, raw).values():
        view.copy_(rand(view.shape, view.dtype))
    window_shape = (
        (LAYERS, ROWS, CONV_DIM, WIDTH + CONV_KM1 - 1)
        if dedup
        else (LAYERS, ROWS, WIDTH, CONV_DIM, CONV_KM1)
    )
    tensors = dict(
        window=rand(window_shape, conv_dtype),
        replayssm_d=rand((LAYERS, ROWS, HV, RING, D), torch.bfloat16, 0.05),
        replayssm_k=rand((LAYERS, ROWS, H, RING, D), torch.bfloat16, 0.09),
        replayssm_g=-torch.rand(LAYERS, ROWS, HV, RING, device=DEVICE, generator=gen)
        * 0.5,
        replayssm_rawv=rand((LAYERS, ROWS, HV, RING, D), torch.bfloat16, 1e-4),
        replayssm_rawk=rand((LAYERS, ROWS, H, RING, D), torch.bfloat16, 1e-4),
        write_pos=torch.zeros(ROWS, dtype=torch.int32, device=DEVICE),
        cache_base=torch.zeros(ROWS, dtype=torch.int32, device=DEVICE),
        is_flush=torch.zeros(ROWS, dtype=torch.int8, device=DEVICE),
    )
    for part in geometry.scratch_parts:
        tensors[part.name] = rand(part.tensor_shape(ROWS), part.dtype)
    return SimpleNamespace(
        geometry=geometry, spec=spec, raw=raw, tensors=tensors, dedup=dedup
    )


class _Arm:
    """One commit over its own copy of every tensor of a case."""

    def __init__(self, case):
        from sglang.srt.mem_cache.kv.layout.state import NGramView, ShortConvView
        from sglang.srt.mem_cache.memory_pool import MambaPool
        from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART

        self.raw = case.raw.clone()
        self.tensors = {name: t.clone() for name, t in case.tensors.items()}
        self.views = _views(case.spec, self.raw)
        t = self.tensors
        self.state = MambaPool.SpeculativeState(
            conv=[self.views["conv.0"]],
            temporal=self.views["temporal"],
            intermediate_ssm=None,
            intermediate_conv_window=[_window(t["window"], case.dedup)],
            **{k: t[k] for k in t if k.startswith("replayssm_")},
        )
        parts = {
            SHORT_CONV_PART: ShortConvView(case.geometry.extra_parts, self.views, t),
            NGRAM_PART: NGramView(case.geometry.extra_parts, self.views, t),
        }
        self.router = SimpleNamespace(part_view=parts.get)


def _rows(*, track: bool, src: bool, steps=STEP_ROWS):
    def dev(values, dtype):
        return torch.tensor(values, dtype=dtype, device=DEVICE)

    # As served: int32 working slots and accept lengths, int64 track slots and steps.
    # The accept lengths stay those of STEP_ROWS, so a perturbed step moves only the rollback.
    return dict(
        slots=dev(SLOT_ROWS, torch.int32),
        ring=dev(list(range(len(SLOT_ROWS))), torch.int64),
        accept=dev([min(max(s, 0), WIDTH - 1) + 1 for s in STEP_ROWS], torch.int32),
        steps=dev(steps, torch.int64),
        track=dev(TRACK_ROWS, torch.int64) if track else None,
        track_steps=dev(TRACK_STEP_ROWS, torch.int64) if track else None,
        src=dev(SRC_ROWS, torch.int64) if src else None,
    )


def _commit(arm, rows, *, multi: bool, ple_only: bool = False):
    """The circular GDN commit and the PLE rollback; returns how many multi-pair launches ran."""
    gdn, scatter, hla = _modules()
    t = arm.tensors
    with (
        mock.patch.object(gdn, "_COMMIT_MULTI_SCATTER", multi),
        mock.patch.object(hla, "_COMMIT_MULTI_SCATTER", multi),
        mock.patch.object(
            scatter,
            "fused_conv_window_scatter_multi",
            wraps=scatter.fused_conv_window_scatter_multi,
        ) as conv_multi,
        mock.patch.object(
            hla, "fused_state_scatter_multi", wraps=hla.fused_state_scatter_multi
        ) as ple_multi,
    ):
        if not ple_only:
            gdn.commit_gdn_replayssm_after_verify(
                spec_state=arm.state,
                write_pos=t["write_pos"],
                cache_base=t["cache_base"],
                is_flush=t["is_flush"],
                state_batch_indices=rows["slots"],
                replay_indices=rows["ring"],
                accept_lens=rows["accept"],
                last_correct_step_indices=rows["steps"],
                fold_every_commit=True,
                mamba_track_indices=rows["track"],
                mamba_steps_to_track=rows["track_steps"],
                src_indices_raw=rows["src"],
                null_block_id=-1,
            )
        hla.commit_ple_states_after_verify(
            arm.router,
            rows["slots"],
            rows["steps"],
            rows["track"],
            rows["track_steps"],
            rows["src"],
        )
    torch.cuda.synchronize()
    return conv_multi.call_count, ple_multi.call_count


def _bits(t):
    return t.view(
        {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()
        ]
    )


def _assert_same(a, b):
    assert torch.equal(a.raw, b.raw), "state arena"
    for name in a.tensors:
        assert torch.equal(_bits(a.tensors[name]), _bits(b.tensors[name])), name


def _row_sets(track: bool):
    sets = [(SLOT_ROWS, STEP_ROWS)]
    return sets + [(TRACK_ROWS, TRACK_STEP_ROWS)] if track else sets


def _side_state(views, name, slot):
    return views[name][slot] if name == "ngram" else _bits(views[name][:, slot])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("track", [True, False])
@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("src", [False, True])
def test_commit_matches_stock(seed, track, dedup, conv_dtype, src):
    """The whole verify commit with the switch on against the switch off: arena, scratch, rings and cursors bit for
    bit. BF16 conv windows without source rows take the multi-pair conv launch, and the PLE side states the multi-pair
    state launch; FP32 conv windows and PP source rows keep the per-pair launches (on rows cast once)."""
    case = _case(seed, conv_dtype=conv_dtype, dedup=dedup)
    rows = _rows(track=track, src=src)
    stock = _Arm(case)
    assert _commit(stock, rows, multi=False) == (0, 0)
    multi = _Arm(case)
    launched = _commit(multi, rows, multi=True)
    assert launched == (int(conv_dtype == torch.bfloat16 and not src), int(not src))
    _assert_same(stock, multi)

    # Slots no row names keep every byte; the side states move only for rows whose step is in the verify window.
    entry = case.spec.entry_bytes()
    before, after = case.raw.view(SLOTS, entry), stock.raw.view(SLOTS, entry)
    named = {s for slots, _ in _row_sets(track) for s in slots if s >= 0}
    for slot in set(range(SLOTS)) - named:
        assert torch.equal(before[slot], after[slot]), slot
    initial = _views(case.spec, case.raw)
    for slots, steps in _row_sets(track):
        for slot, step in zip(slots, steps):
            if slot < 0:
                continue
            for name in ("conv.0", "short_conv", "ngram"):
                moved = not torch.equal(
                    _side_state(initial, name, slot),
                    _side_state(stock.views, name, slot),
                )
                assert moved == (0 <= step < WIDTH), (name, slot, step)

    # Control: one valid row's accepted step perturbed, fed to the switched-on commit only, must move every side state.
    wrong_steps = list(STEP_ROWS)
    wrong_steps[0] += 1
    other = _Arm(case)
    _commit(other, _rows(track=track, src=src, steps=wrong_steps), multi=True)
    for name in ("conv.0", "short_conv", "ngram"):
        assert not torch.equal(
            _side_state(other.views, name, SLOT_ROWS[0]),
            _side_state(multi.views, name, SLOT_ROWS[0]),
        ), name


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn]
)
@pytest.mark.parametrize("track", [True, False])
def test_ple_rollback_matches_stock_for_every_dtype(dtype, track):
    """The PLE short-conv state takes the conv dtype; whatever it is, beside the int64 n-gram history, the multi-pair
    launch moves the per-pair launches' bytes."""
    _, scatter, _ = _modules()
    case = _case(3, conv_dtype=dtype)
    rows = _rows(track=track, src=False)
    stock, multi = _Arm(case), _Arm(case)
    assert _commit(stock, rows, multi=False, ple_only=True) == (0, 0)
    assert _commit(multi, rows, multi=True, ple_only=True) == (0, 1)
    _assert_same(stock, multi)
    assert not torch.equal(stock.raw, case.raw)
    pairs = [
        (multi.views["short_conv"], multi.tensors["intermediate_short_conv"]),
        (
            multi.views["ngram"].unsqueeze(0),
            multi.tensors["intermediate_ngram"].unsqueeze(0),
        ),
    ]
    assert (
        scatter._state_multi_word(pairs) == 8
    )  # the declared layout aligns every part


@pytest.mark.parametrize("track", [True, False])
def test_launch_counts(track):
    """Per verify round, as served (int32 working slots, int64 track slots and steps): the stock commit launches the 3
    GDN commit kernels plus, per conv type and PLE side state and per row set, a scatter, its step cast, its
    source-row arange and, for int64 track slots, a slot cast; with the switch on, the 3 commit kernels and 2 scatters."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    case = _case(0)
    rows = _rows(track=track, src=False)
    counts = {}
    for multi in (False, True):
        arm = _Arm(case)
        _commit(arm, rows, multi=multi)  # compiles, and builds the pointer tables
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _commit(arm, rows, multi=multi)
        kernels = [
            e.name
            for e in prof.events()
            if e.device_type == DeviceType.CUDA
            and not e.name.startswith(("Memcpy", "Memset"))
        ]
        counts[multi] = (len(kernels), kernels)
    stock = 3 + (3 + 4) * 3 if track else 3 + 3 * 3
    assert counts[False][0] == stock, counts[False][1]
    assert counts[True][0] == 3 + 2, counts[True][1]
