"""Microbench: current DS score+selector vs tsinghua-ideal/flash-topk-attention.

Compares four paths at DS-relevant shapes, all reading the same scored
input where applicable so the comparison is apples-to-apples:

  P1. ``score_triton + torch.topk + build_selected_physical``
      The current production path. Baseline.

  P2. ``score_triton + flashinfer.top_k_page_table_transform + append_sink_recent``
      The existing optional selector. Wired but currently capture-unsafe;
      this re-measures both per-call cost and graph-capture status.

  P3. ``score_triton + ftka.cuda_ops.raft_topk + build_selected_physical``
      Substitutes ``torch.topk`` with FTKA's RAFT radix top-k. Same
      logical->physical Triton kernel as P1.

  P4. ``ftka.cuda_ops.batched_sparse_gemv (score) + ftka.cuda_ops.raft_topk
       + build_selected_physical``
      Substitutes BOTH the score kernel and the top-k step with FTKA.
      Requires a one-time paged-cache layout transform of our K_label
      cache; per the plan, that setup cost is reported alongside the
      per-call timing.

Per shape (bs, h_kv, max_ctx, top_k) we record:

  * ``mean_us``         per-call selector(+score for P4) wall-clock
  * ``parity_match``    bool — selected physical id sets equal P1 per
                        ``(bs, h_kv)`` (set equality; within-top-k order
                        differences are tolerated)
  * ``graph_status``    one of ``"ok"`` / ``"capture_fail:<reason>"`` /
                        ``"replay_fail:<reason>"``
  * ``score_us``        per-call score-only wall-clock (only meaningful
                        for P4; reported as the Triton score baseline
                        on P1/P2/P3)

FTKA is treated as optional. Paths P3 and P4 emit a single ``SKIPPED``
record (with the import error message) when ``ftka`` is not installed —
the script still produces a complete P1/P2 table.

Output:

  * ``benchmark/double_sparsity/repro_session/ftka_comparison/results.json``
  * ``benchmark/double_sparsity/repro_session/ftka_comparison/results.md``

Usage:

  PYTHONPATH=python python3 \\
      benchmark/double_sparsity/repro_session/microbench_ftka_backends.py \\
      --quick    # smaller grid for smoke testing

Pinned FTKA commit: ``d8803b29961c44d77a747636ad4282bd7a9094af``. Mismatch
between the installed ftka version and the pinned target is recorded in
the JSON metadata but does not abort the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

# --------------------------------------------------------------------------- #
# Optional FTKA detection                                                     #
# --------------------------------------------------------------------------- #

_FTKA_TARGET_COMMIT_FALLBACK = "d8803b29961c44d77a747636ad4282bd7a9094af"


def _detect_ftka() -> tuple[bool, str, str]:
    """Returns (available, version_string, reason_if_unavailable)."""
    if importlib.util.find_spec("ftka") is None:
        return False, "", "ftka package not installed"
    try:
        import ftka  # noqa: F401

        version = getattr(ftka, "__version__", "<unknown>")
        return True, version, ""
    except Exception as e:  # pragma: no cover - defensive: avoid import crash
        return False, "", f"ftka import raised: {e!r}"


def _detect_flashinfer() -> tuple[bool, str]:
    if importlib.util.find_spec("flashinfer") is None:
        return False, "flashinfer not installed"
    try:
        import flashinfer

        return True, getattr(flashinfer, "__version__", "<unknown>")
    except Exception as e:
        return False, f"flashinfer import raised: {e!r}"


# --------------------------------------------------------------------------- #
# Shape grid                                                                  #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Shape:
    bs: int
    h_kv: int
    max_ctx: int
    top_k: int
    sink: int = 4
    recent: int = 64
    seq_len: int | None = None  # default = max_ctx - 1 (near-full history)

    @property
    def effective_seq_len(self) -> int:
        return self.seq_len if self.seq_len is not None else self.max_ctx - 1


def default_grid() -> list[Shape]:
    """Plan-spec grid: ctx in {32K, 64K, 128K}; bs in {1, 4, 8, 16, 32};
    top_k in {512, 1024, 2048, 4096, 8192}; h_kv=1. Plus one h_kv=8
    stress row."""
    shapes: list[Shape] = []
    for ctx in (32768, 65536, 131072):
        for bs in (1, 4, 8, 16, 32):
            for top_k in (512, 1024, 2048, 4096, 8192):
                shapes.append(Shape(bs=bs, h_kv=1, max_ctx=ctx, top_k=top_k))
    # Stress: h_kv=8 at a mid-sized shape.
    shapes.append(Shape(bs=4, h_kv=8, max_ctx=65536, top_k=2048))
    return shapes


def quick_grid() -> list[Shape]:
    shapes: list[Shape] = []
    for bs in (1, 16, 32):
        for top_k in (1024, 2048):
            shapes.append(Shape(bs=bs, h_kv=1, max_ctx=131072, top_k=top_k))
    shapes.append(Shape(bs=4, h_kv=8, max_ctx=65536, top_k=2048))
    return shapes


# --------------------------------------------------------------------------- #
# Test-input fixture                                                          #
# --------------------------------------------------------------------------- #


def make_inputs(shape: Shape, device: torch.device, seed: int = 0):
    """Synthetic score buffer + identity req_to_token. Matches the layout
    the production score kernel emits: sink/recent/oob masked to -inf."""
    torch.manual_seed(seed)
    att_out = torch.randn(
        shape.bs, shape.h_kv, shape.max_ctx, dtype=torch.float32, device=device
    )
    att_out[..., : shape.sink] = float("-inf")
    seq_len = shape.effective_seq_len
    att_out[..., seq_len - shape.recent : seq_len] = float("-inf")
    att_out[..., seq_len - 1 :] = float("-inf")
    r2t = (
        torch.arange(shape.max_ctx, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(shape.bs, shape.max_ctx)
        .contiguous()
    )
    seq_lens = torch.full((shape.bs,), seq_len, dtype=torch.int64, device=device)
    total = shape.top_k + shape.sink + shape.recent
    out = torch.zeros((shape.bs, shape.h_kv, total), dtype=torch.int32, device=device)
    return att_out, r2t, seq_lens, out


def extract_topk_set(out: torch.Tensor, shape: Shape) -> list[set[int]]:
    """Return the SET of selected top-k physical ids per (bs, h_kv) row,
    flattened into bs*h_kv sets. Sink / recent slots are deterministic
    across backends; only the top-k slot is set-compared."""
    sets: list[set[int]] = []
    for b in range(shape.bs):
        for h in range(shape.h_kv):
            sets.append(set(out[b, h, : shape.top_k].cpu().tolist()))
    return sets


# --------------------------------------------------------------------------- #
# CUDA-graph capture probe                                                    #
# --------------------------------------------------------------------------- #


def probe_graph(fn: Callable[[], None], n_warmup: int = 3) -> tuple[str, str]:
    """Try to capture+replay fn() under a CUDA graph. Returns
    ``(status, detail)`` where status is one of ``"ok"``, ``"capture_fail"``,
    ``"replay_fail"``."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            try:
                fn()
            except Exception as e:
                return "warmup_fail", str(e)[:200]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception as e:
        return "capture_fail", str(e)[:200]
    try:
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:
        return "replay_fail", str(e)[:200]
    return "ok", ""


# --------------------------------------------------------------------------- #
# Path runners                                                                #
# --------------------------------------------------------------------------- #


class _PathRunner:
    """A path-runner wraps one (score + selector) configuration so the
    timing/parity/graph probes share the same harness."""

    name: str = "<base>"

    def setup(
        self, shape: Shape, device: torch.device
    ) -> tuple[Any, Callable[[], None]]:
        """Return (state, run_callable). ``state`` carries any tensors the
        caller needs for parity extraction. ``run_callable`` performs one
        full step (score → selector → fill ``state.out``)."""
        raise NotImplementedError

    def extract_output(self, state: Any) -> torch.Tensor:
        return state["out"]


def _launch_score(att_out, q_label, k_label, r2t, seq_lens, shape, block_t=1024):
    """Thin wrapper around the production score kernel so all paths share
    the same score-phase cost when they use it."""
    from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
        _launch_score as _ds_launch_score,
    )

    _ds_launch_score(
        q_label=q_label,
        k_label_layer=k_label,
        req_to_token_indexed=r2t,
        seq_lens=seq_lens,
        att_out=att_out,
        sm_scale=1.0,
        sink_tokens=shape.sink,
        recent_tokens=shape.recent,
        block_t=block_t,
    )


def _score_fixture(shape: Shape, device: torch.device, seed: int = 0):
    """Build Q_label + K_label so we can exercise the real Triton score
    kernel in each path. ``S`` (heavy-channel count) hardcoded to 32 per
    the plan-spec ``heavy=32``. K-label pool sized to max_ctx tokens."""
    torch.manual_seed(seed)
    s = 32
    q_label = torch.randn(shape.bs, shape.h_kv, s, dtype=torch.float32, device=device)
    k_label = torch.randn(
        shape.max_ctx, shape.h_kv, s, dtype=torch.bfloat16, device=device
    )
    return q_label, k_label


class _SelectorOnlyRunner(_PathRunner):
    """Selector-only path: caller pre-runs the Triton score and freezes
    the score buffer, so we time just the top-k+build step. This matches
    the structure of ``microbench_selector_backends.py`` so timings line
    up with the existing baseline."""

    def __init__(self, backend: str):
        self.name = backend

    def setup(self, shape, device):
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            make_selector,
        )

        att_out, r2t, seq_lens, out = make_inputs(shape, device)
        # Run the score Triton kernel once to populate `att_out`; for
        # selector-only paths the score cost is amortized away by being
        # outside the timed region. (The "score_us" column is reported
        # separately so users can add it back.)
        q_label, k_label = _score_fixture(shape, device)
        _launch_score(att_out, q_label, k_label, r2t, seq_lens, shape)
        # Re-mask sink/recent/oob after the score writes (paranoia in
        # case the synthetic Q/K dot doesn't produce the same -inf
        # pattern as the production kernel). The score kernel above
        # ALREADY masks via the sink/recent args we pass, so this is a
        # belt-and-braces.
        seq_len = shape.effective_seq_len
        att_out[..., : shape.sink] = float("-inf")
        att_out[..., seq_len - shape.recent : seq_len] = float("-inf")
        att_out[..., seq_len - 1 :] = float("-inf")

        selector = make_selector(
            self.name,
            max_bs=shape.bs,
            h_kv=shape.h_kv,
            device=device,
            max_top_k=shape.top_k,
            max_ctx=shape.max_ctx,
        )

        def run():
            selector.select(
                att_out_approx=att_out,
                req_to_token_indexed=r2t,
                seq_lens=seq_lens,
                top_k=shape.top_k,
                sink_tokens=shape.sink,
                recent_tokens=shape.recent,
                out=out,
            )

        state = {"out": out, "att_out": att_out, "r2t": r2t, "seq_lens": seq_lens}
        return state, run


class _FtkaScoreAndSelectRunner(_PathRunner):
    """Path 4: substitute the score kernel with ``ftka.cuda_ops.
    batched_sparse_gemv`` AND the top-k with ``ftka.cuda_ops.raft_topk``.

    FTKA's GEMV consumes a paged KV cache (kv_indices / kv_indptr /
    kv_last_page_len) over a ``[T_pool, H_kv, head_dim]`` k buffer. Our
    K-label cache shape is ``[T_pool, H_kv, S=32]`` which is compatible
    when we view ``head_dim := S``. To bridge, the runner builds a
    page_size=1 paged view: ``kv_indices = req_to_token_indexed.flatten()``,
    ``kv_indptr = arange(bs+1) * seq_len``, ``kv_last_page_len = ones(bs)``.
    Building those buffers happens in ``setup()``, *not* on the hot path,
    so the per-call timing reflects only the FTKA kernels.

    A separate metadata field (``layout_transform_us``) records the
    one-time cost of building the page-table view, so a reader can decide
    if amortizing the transform across many steps is realistic in a
    server deployment (it isn't in general — ``req_to_token`` changes
    every step).
    """

    name = "ftka_gemv+ftka_topk"

    def setup(self, shape, device):
        from ftka.cuda_ops import batched_sparse_gemv as _gemv  # type: ignore
        from ftka.cuda_ops import raft_topk as _raft_topk  # type: ignore

        q_label, k_label = _score_fixture(shape, device)
        _, r2t, seq_lens, out = make_inputs(shape, device)

        # Build the page-table view once. Page size = 1: each token is
        # one page, kv_indices is just req_to_token flattened.
        t_xform = time.perf_counter()
        seq_len = shape.effective_seq_len
        kv_indices = r2t[:, :seq_len].reshape(-1).to(torch.int32)
        kv_indptr = (
            torch.arange(shape.bs + 1, dtype=torch.int32, device=device) * seq_len
        )
        kv_last_page_len = torch.ones(shape.bs, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        xform_us = (time.perf_counter() - t_xform) * 1e6

        # Reusable scratch for top-k + intermediate score.
        num_rows = shape.bs * shape.h_kv
        score_buf = torch.empty(
            (shape.bs, shape.h_kv, shape.max_ctx),
            dtype=torch.float32,
            device=device,
        )
        # Re-mask the slots FTKA won't touch (the gemv only writes
        # active history positions; sink/recent/oob masking is enforced
        # by our downstream raft_topk + build kernel).
        score_buf.fill_(float("-inf"))
        # Active region the FTKA gemv will fill.
        # FTKA's gemv writes to a flat output sized by sum(seq_lens),
        # not [bs, max_ctx]. We allocate a contiguous scoring slab and
        # then scatter into score_buf prior to topk.
        gemv_out = torch.empty(num_rows * seq_len, dtype=torch.float32, device=device)

        # raft_topk scratch.
        values_buf = torch.empty(
            (num_rows, shape.top_k), dtype=torch.float32, device=device
        )
        indices_buf = torch.empty(
            (num_rows, shape.top_k), dtype=torch.int32, device=device
        )
        scratch_buf = torch.empty(
            num_rows * shape.max_ctx, dtype=torch.int32, device=device
        )

        # Preconstruct a scatter map: gemv_out[row, t] -> score_buf[b, h, t].
        # Since seq_lens are uniform here, the layout is contiguous in
        # the active region and a single .view() suffices.
        from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
            _build_selected_physical,
        )

        def run():
            # 1. FTKA paged-cache sparse gemv -> gemv_out.
            _gemv(
                q_label,  # query  [bs, H_kv, S]
                gemv_out,  # output [bs * seq_len]
                k_label,  # k cache [T_pool, H_kv, S]
                kv_indices,
                kv_indptr,
                kv_last_page_len,
            )
            # 2. Scatter into masked score_buf so raft_topk sees -inf on
            # sink/recent/oob.
            score_buf[..., :seq_len].copy_(gemv_out.view(shape.bs, shape.h_kv, seq_len))
            seq_len_pad = shape.effective_seq_len
            score_buf[..., : shape.sink] = float("-inf")
            score_buf[..., seq_len_pad - shape.recent : seq_len_pad] = float("-inf")
            score_buf[..., seq_len_pad - 1 :] = float("-inf")
            # 3. raft_topk -> indices_buf.
            scores_flat = score_buf.view(num_rows, shape.max_ctx)
            _raft_topk(scores_flat, values_buf, indices_buf, scratch_buf, shape.top_k)
            # 4. Reuse the existing _build_selected_physical Triton
            # kernel to produce the final [bs, h_kv, total] output.
            _build_selected_physical(
                topk_logical=indices_buf.view(shape.bs, shape.h_kv, shape.top_k),
                req_to_token_indexed=r2t,
                seq_lens=seq_lens,
                sink_tokens=shape.sink,
                recent_tokens=shape.recent,
                out=out,
            )

        state = {"out": out, "layout_transform_us": xform_us}
        return state, run


# --------------------------------------------------------------------------- #
# Timing harness                                                              #
# --------------------------------------------------------------------------- #


def time_callable(
    run: Callable[[], None], n_warmup: int = 10, n_iters: int = 50
) -> float:
    """Returns mean us / call."""
    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6


def time_score_only(shape: Shape, device: torch.device) -> float:
    """Score-kernel baseline: time only ``_launch_score`` so the markdown
    table can show "score_us" alongside each selector's per-call cost."""
    att_out, r2t, seq_lens, _ = make_inputs(shape, device)
    q_label, k_label = _score_fixture(shape, device)

    def run():
        _launch_score(att_out, q_label, k_label, r2t, seq_lens, shape)

    return time_callable(run)


# --------------------------------------------------------------------------- #
# Main sweep                                                                  #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class PathResult:
    path: str
    shape: dict[str, int]
    status: str  # "ok" / "skipped" / "error"
    mean_us: float | None = None
    score_us: float | None = None
    parity_match: bool | None = None
    graph_status: str | None = None
    graph_detail: str | None = None
    layout_transform_us: float | None = None
    error: str | None = None


def run_one(
    runner: _PathRunner,
    shape: Shape,
    device: torch.device,
    baseline_topk_sets: list[set[int]] | None,
) -> PathResult:
    shape_dict = dataclasses.asdict(shape)
    try:
        state, run = runner.setup(shape, device)
    except Exception as e:
        return PathResult(
            path=runner.name,
            shape=shape_dict,
            status="error",
            error=f"setup: {type(e).__name__}: {e}",
        )

    try:
        mean_us = time_callable(run)
    except Exception as e:
        return PathResult(
            path=runner.name,
            shape=shape_dict,
            status="error",
            error=f"run: {type(e).__name__}: {e}",
        )

    parity_match: bool | None = None
    if baseline_topk_sets is not None:
        # Run once more to ensure `out` reflects the latest call.
        run()
        torch.cuda.synchronize()
        out = state["out"]
        sets = extract_topk_set(out, shape)
        parity_match = all(sets[i] == baseline_topk_sets[i] for i in range(len(sets)))

    graph_status, graph_detail = probe_graph(run)

    return PathResult(
        path=runner.name,
        shape=shape_dict,
        status="ok",
        mean_us=mean_us,
        parity_match=parity_match,
        graph_status=graph_status,
        graph_detail=graph_detail,
        layout_transform_us=state.get("layout_transform_us"),
    )


def baseline_topk_sets(shape: Shape, device: torch.device) -> list[set[int]]:
    runner = _SelectorOnlyRunner("torch")
    state, run = runner.setup(shape, device)
    run()
    torch.cuda.synchronize()
    return extract_topk_set(state["out"], shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="reduced grid")
    parser.add_argument(
        "--output-dir",
        default="benchmark/double_sparsity/repro_session/ftka_comparison",
        help="directory for results.json + results.md",
    )
    parser.add_argument(
        "--shape-limit",
        type=int,
        default=None,
        help="only run the first N shapes (debugging)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this microbench requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    shapes = quick_grid() if args.quick else default_grid()
    if args.shape_limit is not None:
        shapes = shapes[: args.shape_limit]

    fi_avail, fi_version = _detect_flashinfer()
    ftka_avail, ftka_version, ftka_skip_reason = _detect_ftka()
    ftka_installed_commit = ""
    if ftka_avail:
        ftka_installed_commit = getattr(
            importlib.import_module("ftka"), "__commit__", "<unknown>"
        )

    print(f"# FTKA microbench — {len(shapes)} shapes")
    print(f"# torch: {torch.__version__}")
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# flashinfer: {'OK ' + fi_version if fi_avail else 'SKIP — ' + fi_version}")
    print(
        f"# ftka: {'OK ' + ftka_version if ftka_avail else 'SKIP — ' + ftka_skip_reason}"
    )
    print(f"# ftka target commit (pinned): {_FTKA_TARGET_COMMIT_FALLBACK}")
    print(f"# ftka installed commit: {ftka_installed_commit}")
    print()

    results: list[PathResult] = []
    from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
        FLASHINFER_TOPK_MAX,
    )

    for i, shape in enumerate(shapes):
        print(
            f"[{i+1}/{len(shapes)}] bs={shape.bs} h_kv={shape.h_kv} "
            f"ctx={shape.max_ctx} top_k={shape.top_k}"
        )
        # P1 baseline (also produces the parity oracle).
        try:
            base_sets = baseline_topk_sets(shape, device)
        except Exception as e:
            print(f"  P1 baseline setup error: {e}")
            results.append(
                PathResult(
                    path="torch",
                    shape=dataclasses.asdict(shape),
                    status="error",
                    error=str(e),
                )
            )
            continue

        results.append(run_one(_SelectorOnlyRunner("torch"), shape, device, base_sets))

        # P2: FlashInfer top_k_page_table — skipped above ceiling.
        if not fi_avail:
            results.append(
                PathResult(
                    path="flashinfer_topk_page_table",
                    shape=dataclasses.asdict(shape),
                    status="skipped",
                    error=fi_version,
                )
            )
        elif shape.top_k > FLASHINFER_TOPK_MAX:
            results.append(
                PathResult(
                    path="flashinfer_topk_page_table",
                    shape=dataclasses.asdict(shape),
                    status="skipped",
                    error=f"top_k>{FLASHINFER_TOPK_MAX} ceiling",
                )
            )
        else:
            results.append(
                run_one(
                    _SelectorOnlyRunner("flashinfer_topk_page_table"),
                    shape,
                    device,
                    base_sets,
                )
            )

        # P3: FTKA raft_topk only.
        if not ftka_avail:
            results.append(
                PathResult(
                    path="ftka_raft_topk",
                    shape=dataclasses.asdict(shape),
                    status="skipped",
                    error=ftka_skip_reason,
                )
            )
        else:
            results.append(
                run_one(
                    _SelectorOnlyRunner("ftka_raft_topk"),
                    shape,
                    device,
                    base_sets,
                )
            )

        # P4: FTKA gemv + FTKA raft_topk.
        if not ftka_avail:
            results.append(
                PathResult(
                    path="ftka_gemv+ftka_topk",
                    shape=dataclasses.asdict(shape),
                    status="skipped",
                    error=ftka_skip_reason,
                )
            )
        else:
            results.append(
                run_one(_FtkaScoreAndSelectRunner(), shape, device, base_sets)
            )

        # Score baseline (informational; same for all selector-only paths).
        score_us = time_score_only(shape, device)
        for r in results[-4:]:
            if r.path != "ftka_gemv+ftka_topk":
                r.score_us = score_us

    # ----------------------------------------------------------------- #
    # Emit                                                              #
    # ----------------------------------------------------------------- #

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "flashinfer": {"available": fi_avail, "version": fi_version},
        "ftka": {
            "available": ftka_avail,
            "version": ftka_version,
            "installed_commit": ftka_installed_commit,
            "target_commit": _FTKA_TARGET_COMMIT_FALLBACK,
            "skip_reason": ftka_skip_reason,
        },
        "n_shapes": len(shapes),
    }

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {"meta": meta, "results": [dataclasses.asdict(r) for r in results]},
            indent=2,
        )
    )
    print(f"\nwrote {json_path}")

    md_path = out_dir / "results.md"
    md_path.write_text(_format_markdown(meta, results))
    print(f"wrote {md_path}")


def _format_markdown(meta: dict, results: list[PathResult]) -> str:
    lines: list[str] = []
    lines.append("# FTKA vs current DS microbench results\n")
    lines.append("## Environment\n")
    lines.append(f"- torch: `{meta['torch_version']}`")
    lines.append(f"- device: `{meta['device']}`")
    lines.append(f"- platform: `{meta['platform']}`")
    fi = meta["flashinfer"]
    lines.append(
        f"- flashinfer: {'OK `' + fi['version'] + '`' if fi['available'] else 'SKIP — ' + fi['version']}"
    )
    ftka = meta["ftka"]
    if ftka["available"]:
        lines.append(
            f"- ftka: OK `{ftka['version']}` "
            f"(installed_commit=`{ftka['installed_commit']}`, "
            f"target_commit=`{ftka['target_commit']}`)"
        )
    else:
        lines.append(
            f"- ftka: SKIP — `{ftka['skip_reason']}` "
            f"(target_commit=`{ftka['target_commit']}`)"
        )
    lines.append("")

    # Group by shape, list paths.
    lines.append("## Per-shape per-path results\n")
    lines.append(
        "| bs | h_kv | ctx | top_k | path | status | mean µs | score µs | "
        "parity | graph | extra |"
    )
    lines.append(
        "|---:|-----:|----:|------:|------|--------|--------:|---------:|"
        "--------|-------|-------|"
    )
    for r in results:
        s = r.shape
        mean_us = f"{r.mean_us:.1f}" if r.mean_us is not None else "-"
        score_us = f"{r.score_us:.1f}" if r.score_us is not None else "-"
        parity = (
            "ok"
            if r.parity_match is True
            else "fail" if r.parity_match is False else "-"
        )
        graph = r.graph_status or "-"
        extra_bits: list[str] = []
        if r.layout_transform_us is not None:
            extra_bits.append(f"xform={r.layout_transform_us:.0f}µs")
        if r.error and r.status != "ok":
            extra_bits.append(r.error[:60])
        if r.graph_detail and r.graph_status not in (None, "ok"):
            extra_bits.append(r.graph_detail[:60])
        extra = "; ".join(extra_bits)
        lines.append(
            f"| {s['bs']} | {s['h_kv']} | {s['max_ctx']} | {s['top_k']} | "
            f"`{r.path}` | {r.status} | {mean_us} | {score_us} | {parity} | "
            f"{graph} | {extra} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
