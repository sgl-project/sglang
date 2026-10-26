"""Muse Spark static expert placement and routing recorder (gfx950).

Permuting a layer's router columns, ``expert_bias`` and expert stacks the way the loader does selects the same
checkpoint experts and produces bitwise the same per-(token, slot) expert rows and full combine; only the EP split
of the combine (which rank's BF16 partial holds which slot) changes.
"""

import numpy as np
import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

MULT, SCALE, EPS = 1.5, 2.5, 1e-6
EP, E = 4, 256


def _perm(seed):
    return torch.randperm(E, generator=torch.Generator().manual_seed(seed))


def _routing_inputs(m):
    torch.manual_seed(m)
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    router_w = torch.randn(8192, E, device="cuda") * 0.02
    pooler_w = (torch.randn(4096, 8192, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda") * 0.01
    norm_w = (torch.rand(4096, device="cuda") + 0.5).to(torch.bfloat16)
    return x, router_w, pooler_w, bias, norm_w


def _routing_fronts(x, router_w, pooler_w, bias, norm_w):
    """(scores, ids, pooled) of every decode / verify routing kernel that serves ``x``'s row count."""
    from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_rows
    from sglang.kernels.ops.moe.router_pooler_topk import (
        joint_router_pooler_applies,
        joint_router_pooler_fp8_applies,
        joint_router_pooler_topk,
        joint_router_pooler_topk_fp8,
        pack_joint_router_pooler_weight,
        quantize_joint_pooler,
        router_pooler_topk,
    )

    args = (bias, norm_w, MULT, SCALE, EPS)
    fronts = {
        "shared": router_pooler_topk(x, router_w, pooler_w, *args, pooler_rowmajor=True)
    }
    packed = pack_joint_router_pooler_weight(router_w, pooler_w)
    if joint_router_pooler_applies(x, packed):
        fronts["joint"] = joint_router_pooler_topk(x, packed, *args)
    pooler_fp8 = quantize_joint_pooler(packed)
    xq, xs = quantize_rows(x)
    if joint_router_pooler_fp8_applies(xq, packed, pooler_fp8[0]):
        fronts["joint_fp8"] = joint_router_pooler_topk_fp8(
            xq, xs, x, packed, pooler_fp8, *args
        )
    return fronts


@pytest.mark.parametrize("m", [1, 8, 16, 24, 32, 64])
def test_placement_keeps_decode_routing(m):
    """Slot ids from the permuted router / bias name the identity run's experts, with bitwise equal weights."""
    from sglang.srt.rocm_muse_spark.expert_placement import (
        permute_expert_vector,
        permute_router_weight,
    )

    x, router_w, pooler_w, bias, norm_w = _routing_inputs(m)
    perm = _perm(m)
    ref = _routing_fronts(x, router_w, pooler_w, bias, norm_w)
    got = _routing_fronts(
        x,
        permute_router_weight(router_w, perm),
        pooler_w,
        permute_expert_vector(bias, perm),
        norm_w,
    )
    assert ref.keys() == got.keys()
    for name in ref:
        (rs, rid, rp), (gs, gid, gp) = ref[name], got[name]
        assert torch.equal(perm.cuda()[gid.long()].int(), rid), name
        assert torch.equal(gs, rs), name
        assert torch.equal(gp, rp), name


@pytest.mark.parametrize("m", [64, 512, 2048])
def test_placement_keeps_prefill_routing(m):
    """The prefill router GEMMs give the identity logits in permuted column order, bit for bit."""
    from sglang.kernels.ops.gemm.router_gemm_fp32 import (
        router_gemm_tuned,
        router_gemv_fp32,
        router_gemv_fp32_supported,
    )
    from sglang.srt.rocm_muse_spark.expert_placement import (
        permute_expert_vector,
        permute_router_weight,
    )

    x, router_w, _, bias, _ = _routing_inputs(m)
    perm = _perm(m)
    placed_w = permute_router_weight(router_w, perm)
    gemms = {"torch": lambda w: x.float() @ w}
    if router_gemv_fp32_supported(x, router_w):
        gemms["gemv"] = lambda w: router_gemv_fp32(x, w)
    if router_gemm_tuned(x, router_w) is not None:
        gemms["tuned"] = lambda w: router_gemm_tuned(x, w)
    placed_bias = permute_expert_vector(bias, perm)
    for name, gemm in gemms.items():
        ref, got = gemm(router_w), gemm(placed_w)
        assert torch.equal(got, ref[:, perm.cuda()]), name
        ref_ids = (torch.sigmoid(ref * MULT) + bias).topk(8, dim=-1).indices
        got_ids = (torch.sigmoid(got * MULT) + placed_bias).topk(8, dim=-1).indices
        assert torch.equal(perm.cuda()[got_ids], ref_ids), name


def _experts(m):
    torch.manual_seed(1000 + m)
    dev = "cuda"
    # Random MXFP4 nibbles and E8M0 scales near 2^0 keep the GEMMs finite.
    stacks = (
        torch.randint(0, 256, (E, 8192, 2048), device=dev, dtype=torch.uint8),
        torch.randint(0, 256, (E, 4096, 2048), device=dev, dtype=torch.uint8),
        torch.randint(118, 124, (E, 8192, 128), device=dev, dtype=torch.uint8),
        torch.randint(118, 124, (E, 4096, 128), device=dev, dtype=torch.uint8),
    )
    x = (torch.randn(m, 4096, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.rand(m, E, device=dev).argsort(-1)[:, :8].int().contiguous()
    scores = torch.rand(m, 8, device=dev)
    norm_w = (torch.rand(4096, device=dev) + 0.5).to(torch.bfloat16)
    return stacks, x, ids, scores, norm_w


@pytest.mark.parametrize("m", [8, 64, 256])
def test_placement_keeps_compact_a4_expert_rows(m):
    """Four fake EP4 ranks: per-slot expert rows and the one-place eight-slot combine are bitwise unchanged."""
    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import compact_a4_experts
    from sglang.srt.rocm_muse_spark.expert_placement import copy_expert_slots
    from sglang.srt.rocm_muse_spark.mxfp4_moe import _finalize_per_slot

    stacks, x, ids, scores, norm_w = _experts(m)
    perm = _perm(m).cuda()
    ids_slot = torch.argsort(perm)[ids.long()].int().contiguous()
    local = E // EP
    rows = {"ref": torch.zeros(m * 8, 4096, dtype=torch.bfloat16, device="cuda")}
    rows["got"] = torch.zeros_like(rows["ref"])
    partial = {"ref": [], "got": []}
    placed = [torch.empty_like(s[:local]) for s in stacks]
    for rank in range(EP):
        start = rank * local
        for target, stack in zip(placed, stacks):
            copy_expert_slots(target, stack, perm[start : start + local])
        for key, rank_ids, weights in (
            ("ref", ids, [s[start : start + local] for s in stacks]),
            ("got", ids_slot, placed),
        ):
            y = compact_a4_experts(x, rank_ids, *weights, local_start=start)
            mine = (rank_ids.flatten() >= start) & (rank_ids.flatten() < start + local)
            rows[key][mine] = y[mine]
            partial[key].append(
                _finalize_per_slot(y, norm_w, scores, rank_ids, start, local, EPS, 1.0)
            )
    assert torch.equal(rows["got"], rows["ref"])
    combined = {
        key: _finalize_per_slot(rows[key], norm_w, scores, i, 0, E, EPS, 1.0)
        for key, i in (("ref", ids), ("got", ids_slot))
    }
    assert torch.equal(combined["got"], combined["ref"])
    # The EP4 sum regroups the slots into other BF16 partials: equal up to their rounding.
    reduced = {
        k: sum(p.float() for p in v).bfloat16().float() for k, v in partial.items()
    }
    error = (reduced["got"] - reduced["ref"]).norm() / reduced["ref"].norm()
    assert error.item() < 4e-3


class _VerifyBatch:
    def __init__(self, valid):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.forward_mode = ForwardMode.TARGET_VERIFY
        self._valid = valid

    def moe_num_token_non_padded(self):
        return self._valid


def test_expert_stats_counts_real_and_padding_rows_in_graph(tmp_path, monkeypatch):
    """Captured record launches count each replay's real / padding rows per slot into its own ring step."""
    from sglang.srt.rocm_muse_spark import expert_stats

    monkeypatch.setattr(expert_stats, "_DRAIN_SECONDS", 0.0)
    monkeypatch.setattr(expert_stats, "_WRITE_SECONDS", 0.0)
    layers, rows = 3, 64
    rec = expert_stats.ExpertStatsRecorder(
        str(tmp_path), layers, E, torch.device("cuda"), header={}
    )
    ids = torch.zeros(layers, rows, 8, dtype=torch.int32, device="cuda")
    valid = torch.zeros(1, dtype=torch.int32, device="cuda")
    batch = _VerifyBatch(valid)
    x = torch.randn(rows, 64, device="cuda").to(torch.bfloat16)
    router = torch.randn(64, E, device="cuda") * 0.1
    bias = torch.zeros(E, device="cuda")

    def forward():
        rec.begin_forward(batch)
        for layer in range(layers):
            rec.record(layer, ids[layer], x, router, bias, MULT)
        rec.end_forward()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    expected = []
    for step, n in enumerate((rows, 17, 0)):
        torch.manual_seed(step)
        ids.copy_(torch.rand(layers, rows, E, device="cuda").argsort(-1)[..., :8])
        valid.fill_(n)
        graph.replay()
        flat = ids.long().cpu()
        expected.append(
            [
                (
                    torch.bincount(flat[l, :n].flatten(), minlength=E),
                    torch.bincount(flat[l, n:].flatten(), minlength=E),
                )
                for l in range(layers)
            ]
        )
    rec.end_forward()  # an eager forward drains the ring
    (path,) = sorted(tmp_path.glob("expert_stats_*_00000.npz"))
    data = np.load(path)
    assert data["step"].tolist() == [0, 1, 2] and int(data["lost"]) == 0
    assert data["meta"][:, :3].tolist() == [[rows, n, 2 + 4] for n in (rows, 17, 0)]
    for step, per_layer in enumerate(expected):
        for layer, (real, pad) in enumerate(per_layer):
            assert data["real"][step, layer].tolist() == real.tolist()
            assert data["pad"][step, layer].tolist() == pad.tolist()
    margin = data["margin"]
    assert (margin[:, 7] == rows + 17).all() and (
        margin[:, :6].sum(1) == rows + 17
    ).all()


def test_expert_stats_margin_flags_ties_and_set_changes(tmp_path):
    """An exact top-8 boundary tie lands in the tie bin; a routed set unlike the FP32 recompute is flagged."""
    from sglang.srt.rocm_muse_spark import expert_stats

    rec = expert_stats.ExpertStatsRecorder(
        str(tmp_path), 1, E, torch.device("cuda"), header={}
    )
    torch.manual_seed(0)
    x = torch.zeros(3, 64, device="cuda", dtype=torch.bfloat16)
    x[1:] = (torch.randn(64, device="cuda") * 0.1).to(torch.bfloat16)
    router = torch.randn(64, E, device="cuda")
    bias = torch.zeros(E, device="cuda")
    top = torch.sigmoid((x.float() @ router) * MULT).topk(9, dim=-1).indices
    # Row 0: every score 0.5, the argmax kernels pick 0..7; row 2 swaps its 8th expert for the 9th.
    ids = torch.stack(
        [torch.arange(8, device="cuda"), top[1, :8], top[2, [0, 1, 2, 3, 4, 5, 6, 8]]]
    )
    rec.begin_forward(
        _VerifyBatch(torch.full((1,), 3, device="cuda", dtype=torch.int32))
    )
    rec.record(0, ids.int().contiguous(), x, router, bias, MULT)
    rec.end_forward()
    margin = rec.margin[0].tolist()
    assert margin[0] == 1 and sum(margin[:6]) == 3
    assert margin[6] == 1 and margin[7] == 3


def test_placement_json_rejects_a_non_permutation(tmp_path):
    from sglang.srt.rocm_muse_spark.expert_placement import ExpertPlacement

    order = list(range(E))
    order[3] = 4
    path = tmp_path / "placement.json"
    path.write_text(f'{{"num_experts": {E}, "layers": {{"5": {order}}}}}')
    with pytest.raises(ValueError, match="not a permutation"):
        ExpertPlacement.from_json(str(path), num_experts=E, ep_size=EP)


def test_loader_places_expert_stacks_router_and_bias(monkeypatch):
    """Rank r's expert slot j, router column 64 r + j and bias entry 64 r + j all hold checkpoint expert
    perm[64 r + j] on the stacked and per-expert checkpoint paths; unlisted layers keep the checkpoint order."""
    from types import SimpleNamespace

    from sglang.srt.rocm_muse_spark import model as muse
    from sglang.srt.rocm_muse_spark.expert_placement import ExpertPlacement

    layer, other = 7, 8
    perm = _perm(layer)
    lm = object.__new__(muse.MuseSparkTextForCausalLM)
    lm.config = SimpleNamespace(num_local_experts=E)
    lm.expert_placement = ExpertPlacement({layer: perm}, num_experts=E)
    ids = torch.arange(E, dtype=torch.float32)
    stack = ids.view(E, 1, 1).expand(E, 2, 3).contiguous()
    prefix = f"model.language_model.layers.{layer}.mlp"
    router = lm._place_routing_tensor(
        f"{prefix}.router.weight", ids.view(1, E).expand(5, E).contiguous()
    )
    bias = lm._place_routing_tensor(f"{prefix}.expert_bias", ids.clone())
    assert torch.equal(router[0], perm.float()) and torch.equal(bias, perm.float())
    local = E // EP
    for rank in range(EP):
        monkeypatch.setattr(
            muse,
            "get_parallel",
            lambda rank=rank: SimpleNamespace(moe_ep_size=EP, moe_ep_rank=rank),
        )
        want = perm[rank * local : (rank + 1) * local].float()
        for lyr, expect in (
            (layer, want),
            (other, ids[rank * local : (rank + 1) * local]),
        ):
            param = torch.empty(local, 2, 3)
            lm._copy_expert_stack(
                param,
                stack,
                f"model.language_model.layers.{lyr}.mlp.experts.w13_weight",
            )
            assert torch.equal(param[:, 0, 0], expect)
        params = {
            f"{prefix}.experts.w13_weight": torch.full((local, 4, 3), -1.0),
            f"{prefix}.experts.w2_weight": torch.full((local, 2, 3), -1.0),
        }
        for e in range(E):
            for proj, value in (
                ("gate_proj", e),
                ("up_proj", e + 1000),
                ("down_proj", e),
            ):
                shape = (2, 3)
                assert lm._load_single_expert(
                    params,
                    f"{prefix}.experts.{e}.{proj}.weight",
                    torch.full(shape, float(value)),
                )
        w13, w2 = (
            params[f"{prefix}.experts.w13_weight"],
            params[f"{prefix}.experts.w2_weight"],
        )
        assert torch.equal(w13[:, 0, 0], want) and torch.equal(
            w13[:, 2, 0], want + 1000
        )
        assert torch.equal(w2[:, 0, 0], want)
