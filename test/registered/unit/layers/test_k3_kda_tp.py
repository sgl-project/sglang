"""Kimi-K3 tensor-parallel KDA under attention DP (--enable-kda-tp).

CPU (gloo, simulated ranks):
- KDA TP applies exactly when --enable-kda-tp is set for Kimi-K3; unsupported features
  (ReplaySSM, HiCache, unified memory, PD disaggregation, the no_buffer radix strategy)
  raise at startup.
- Head-sharded slot ids: each DP rank allocates its own range of the global slot ids.
- Radix prefix hits: every DP rank's deferred copy-on-write and clear ops are applied on
  every rank's head shard, in the same order, when any rank is in extend, and none when
  no rank is; the copy waits for the state layers first.
- Verify commit: every rank scatters every DP rank's accepted states, from the verify
  forward's resolved slots, into its own state pool.
- KimiK3DeltaAttention.forward returns forward_kda_tp's output (already through o_proj).
- The stand-in batches _fast_context (every rank verifies) and _general_context (a rank in
  extend) build carry what KDAAttnBackend.forward_extend reads on target verify, up to
  its first kernel; a ragged verify layout is refused.
"""

import itertools
import os
import queue
import socket
import time
import types
import unittest
import unittest.mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _par(**kw):
    base = dict(
        attn_tp_size=1,
        attn_dp_size=8,
        tp_size=8,
        kda_weight_shard_size=None,
        enable_kda_tp=True,
        attn_dp_rank=0,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestEnable(unittest.TestCase):
    def _applies(self, k3=True, **par):
        from sglang.srt.layers.communication import k3_kda_tp

        with unittest.mock.patch.multiple(
            k3_kda_tp, get_parallel=lambda: _par(**par), _is_kimi_k3=lambda: k3
        ):
            return k3_kda_tp._applies(), k3_kda_tp.state_shard_width()

    def test_on_for_k3(self):
        self.assertEqual(self._applies(), (True, 8))

    def test_off(self):
        self.assertEqual(self._applies(k3=False), (False, 1))
        self.assertEqual(self._applies(enable_kda_tp=False), (False, 1))

    def _check(
        self,
        replayssm=False,
        hicache=False,
        mode="null",
        radix=False,
        unified=False,
        spec="NEXTN",
    ):
        from sglang.srt import runtime_context
        from sglang.srt.layers.communication import k3_kda_tp

        NS = types.SimpleNamespace
        with unittest.mock.patch.multiple(
            runtime_context,
            get_exec=lambda: NS(
                mamba=NS(
                    enable_linear_replayssm_spec=replayssm,
                    enable_linear_replayssm=False,
                    enable_mamba_extra_buffer=True,
                )
            ),
            get_memory=lambda: NS(
                enable_hierarchical_cache=hicache,
                enable_unified_memory=unified,
                disable_radix_cache=not radix,
            ),
            get_spec=lambda: NS(speculative_algorithm=spec),
            get_disagg=lambda: NS(disaggregation_mode=mode),
        ):
            k3_kda_tp._check_supported()

    def test_supported_features(self):
        self._check()
        self._check(radix=True)
        self._check(spec=None)
        self._check(spec="EAGLE")

    def test_unsupported_features_raise(self):
        for kw, msg in (
            (dict(replayssm=True), "ReplaySSM"),
            (dict(hicache=True), "HiCache"),
            (dict(spec="DSPARK"), "speculative algorithm DSPARK"),
            (dict(unified=True), "unified-memory"),
            (dict(mode="prefill"), "PD disaggregation"),
            (dict(mode="decode"), "PD disaggregation"),
        ):
            with self.subTest(kw=kw), self.assertRaisesRegex(NotImplementedError, msg):
                self._check(**kw)


class TestSlots(unittest.TestCase):
    def test_partition_and_allocator(self):
        from sglang.srt.layers.communication import k3_kda_tp
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

        ranges = []
        for rank in range(4):
            with unittest.mock.patch.object(
                k3_kda_tp,
                "get_parallel",
                lambda: _par(tp_size=4, attn_dp_size=4, attn_dp_rank=rank),
            ):
                first, n = k3_kda_tp.slot_partition(41)
            alloc = MambaSlotAllocator(size=n, device="cpu", first_slot=first)
            slots = alloc.alloc(n).tolist()
            self.assertIsNone(alloc.alloc(1))
            ranges.append(slots)
        flat = [s for r in ranges for s in r]
        self.assertEqual(flat, list(range(1, 41)))  # slot 0 stays the padding slot


class _Group:
    """GroupCoordinator stand-in over the default process group."""

    def __init__(self, world):
        self.world_size = world
        self.device_group = None
        self.cpu_group = None

    def all_gather_into_tensor(self, out, x):
        parts = list(out.chunk(self.world_size))
        dist.all_gather(parts, x.contiguous())
        out.copy_(torch.cat(parts))


def _run_workers(target, backend, counts, timeout_sec=600):
    """Run ``target`` on len(counts) spawned ranks; returns (sorted results, exit codes).
    Fails as soon as a rank dies without a result, and never leaves a rank behind (one stuck
    in a collective after a peer failed is terminated, then killed)."""
    world = len(counts)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=target, args=(r, world, port, backend, counts, q))
        for r in range(world)
    ]
    try:
        for p in procs:
            p.start()
        res, deadline = [], time.monotonic() + timeout_sec
        while len(res) < world:
            try:
                res.append(q.get(timeout=5))
            except queue.Empty:
                codes = [p.exitcode for p in procs]
                if (
                    any(c not in (None, 0) for c in codes)
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError(
                        f"{len(res)}/{world} worker results, exit codes {codes}"
                    )
        for p in procs:
            p.join(60)
        return sorted(res), [p.exitcode for p in procs]
    finally:
        started = [p for p in procs if p.pid is not None]
        for p in started:
            if p.is_alive():
                p.terminate()
        for p in started:
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join()


def _mamba_pool_spec():
    from sglang.srt.mem_cache.memory_pool import MambaPool

    return unittest.mock.create_autospec(MambaPool, instance=True)


def _req_pool(mamba_pool, size=0, mapping=None):
    """A HybridReqToTokenPool built without __init__ (the real class's methods and properties
    only) whose state storage holds ``mamba_pool``: a method the class no longer has fails
    here as in production."""
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.size, pool.device = size, "cpu"
    pool.req_index_to_mamba_index_mapping = mapping
    pool.state = StateStorage(
        sub_pool="state",
        pool=mamba_pool,
        slots=None,
        layer_map={},
        short_conv_pool=None,
        ngram_pool=None,
    )
    return pool


def _runner(mamba_pool, router=None, req_pool=None):
    """ModelRunner stand-in: the token allocator's state storage over ``mamba_pool`` and
    the KV pool's layer router."""
    from sglang.srt.mem_cache.kv.state_storage import StateStorage

    state = StateStorage(
        sub_pool="state",
        pool=mamba_pool,
        slots=None,
        layer_map={},
        short_conv_pool=None,
        ngram_pool=None,
    )
    if router is None:
        router = types.SimpleNamespace(part_view=lambda part: mamba_pool)
    return types.SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=types.SimpleNamespace(state=state),
        token_to_kv_pool=types.SimpleNamespace(layer_router=router),
    )


def _cow_worker(rank, world, port, backend, modes, q):
    """Each DP rank brings its own prefix-match COW and clear ops (slots of its global range);
    every rank must apply all of them to its shard, in rank order."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        from sglang.srt.layers.communication import k3_kda_tp

        par = _par(
            tp_size=world, attn_dp_size=world, attn_dp_rank=rank, tp_group=_Group(world)
        )
        state = torch.arange(
            world * 10 + 1, dtype=torch.float32
        )  # one value per global slot
        log = []

        def clear_slots(idx):
            log.append(("clear", idx.tolist()))
            state[idx] = 0

        def copy_from(src, dst):
            log.append(("cow", src.tolist(), dst.tolist()))
            state[dst] = state[src]

        mamba_pool = _mamba_pool_spec()
        mamba_pool.clear_slots.side_effect = clear_slots
        mamba_pool.copy_from.side_effect = copy_from
        mamba_pool.device = "cpu"
        router = types.SimpleNamespace(
            wait_for_part=lambda part: log.append(("wait", part))
        )
        runner = _runner(mamba_pool, router)
        base = rank * 10 + 1
        t = lambda xs: torch.tensor(xs, dtype=torch.int64)
        fb = types.SimpleNamespace(
            dp_forward_modes=[m.value for m in modes],
            mamba_clear_indices=t([base + 9]),
            mamba_cow_src_indices=t([base, base + 1]),
            mamba_cow_dst_indices=t([base + 5, base + 6]),
        )
        with unittest.mock.patch.object(k3_kda_tp, "get_parallel", lambda: par):
            k3_kda_tp.mirror_deferred_state_ops(runner, fb)
        cleared = all(
            getattr(fb, f) is None
            for f in (
                "mamba_clear_indices",
                "mamba_cow_src_indices",
                "mamba_cow_dst_indices",
            )
        )
        q.put((rank, log, state.tolist(), cleared))
    finally:
        dist.destroy_process_group()


class TestRadixDeferredStateOps(unittest.TestCase):
    """Radix prefix hits under KDA TP rely on every rank mirroring the match's COW / clear."""

    def _run(self, modes):
        res, codes = _run_workers(_cow_worker, "gloo", modes)
        self.assertEqual(codes, [0] * len(modes))
        return res

    def test_cow_and_clear_mirrored_on_every_rank(self):
        from sglang.srt.model_executor.forward_mode import ForwardMode

        modes = [
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.IDLE,
            ForwardMode.EXTEND,
        ]
        res = self._run(modes)
        world = len(modes)
        # Like ModelRunner._maybe_execute_deferred_mamba_cow_and_clear: the copy waits
        # for the state part, the clear does not.
        want_log = [
            ("clear", [j * 10 + 10 for j in range(world)]),
            ("wait", "state"),
            (
                "cow",
                [s for j in range(world) for s in (j * 10 + 1, j * 10 + 2)],
                [s for j in range(world) for s in (j * 10 + 6, j * 10 + 7)],
            ),
        ]
        states = set()
        for rank, log, state, cleared in res:
            self.assertEqual(log, want_log, f"rank {rank}")
            self.assertTrue(cleared)
            states.add(tuple(state))
            for j in range(world):
                self.assertEqual(state[j * 10 + 6], j * 10 + 1)
                self.assertEqual(state[j * 10 + 10], 0)
        self.assertEqual(len(states), 1)  # every rank's shard saw the same ops

    def test_no_extend_no_ops(self):
        from sglang.srt.model_executor.forward_mode import ForwardMode

        res = self._run([ForwardMode.DECODE, ForwardMode.IDLE])
        for rank, log, _, _ in res:
            self.assertEqual(log, [], f"rank {rank}")


class TestLayerForward(unittest.TestCase):
    def test_kda_tp_output_bypasses_o_proj(self):
        from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention

        hidden, out, calls = torch.zeros(3, 8), torch.ones(3, 8), []

        def never(name):
            def call(*args, **kwargs):
                raise AssertionError(f"{name} called under KDA TP")

            return call

        attn = torch.nn.Module.__new__(KimiK3DeltaAttention)
        attn.__dict__.update(
            _kda_tp=True,
            forward_kda_tp=lambda x: calls.append(x) or out,
            forward_pre_o_proj=never("forward_pre_o_proj"),
            o_proj=never("o_proj"),
        )
        got = KimiK3DeltaAttention.forward(attn, hidden, None, None, None)
        self.assertIs(got, out)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], hidden)


class _CpuTorch:
    """k3_kda_tp's torch with its CUDA device lookups answered with the CPU."""

    cuda = types.SimpleNamespace(
        current_device=lambda: "cpu", is_current_stream_capturing=lambda: False
    )

    @staticmethod
    def device(*args):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(torch, name)


class _Reached(Exception):
    pass


D, BS, CAP = 2, 2, 4  # draft tokens, requests per rank, per-rank request capacity


def _local_batch(rank, kind):
    """(this rank's ForwardBatch fields, its KDA forward metadata, its row count)."""
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_mode import ForwardMode

    slots = torch.arange(BS, dtype=torch.int32) + rank * CAP + 1
    NS = types.SimpleNamespace
    if kind == "verify":
        mode, rows, lens = ForwardMode.TARGET_VERIFY, BS * D, [D] * BS
        spec, prefix = NS(draft_token_num=D), None
    else:
        mode, rows, lens = ForwardMode.EXTEND, 3, [1, 2]
        spec, prefix = None, torch.tensor([0, 4])
    qsl = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32)
    fm = ForwardMetadata(
        query_start_loc=qsl,
        mamba_cache_indices=slots,
        is_target_verify=kind == "verify",
    )
    fb = NS(
        forward_mode=mode,
        logical_forward_mode=mode,
        batch_size=BS,
        extend_seq_lens_cpu=lens if kind == "extend" else None,
        extend_prefix_lens=prefix,
        spec_info=spec,
        ragged_verify_layout=None,
        mamba_track_mask=None,
    )
    return fb, fm, rows


def _verify_backend(fm):
    """KDAAttnBackend (built without __init__) whose target-verify path stops at its first
    kernel (_Reached), after every forward-batch read on that path."""
    from sglang.srt.layers.attention.linear import kda_backend as kb

    be = object.__new__(kb.KDAAttnBackend)
    be.forward_metadata = fm
    be.req_to_token_pool = types.SimpleNamespace(size=CAP)
    z = torch.zeros(1, 1, 1)
    cache = types.SimpleNamespace(
        conv=[z], temporal=z, intermediate_ssm=z, intermediate_conv_window=[z]
    )
    be.state_layer_cache = lambda layer_id: cache
    be._can_run_dspark_cutedsl_mtp = lambda **kw: False
    be._can_run_fused_chain_verify = lambda **kw: False
    return kb, be


def _run_verify(kb, be, batch, rows):
    def first_kernel(*args, **kw):
        raise _Reached

    layer = types.SimpleNamespace(layer_id=0, conv_weights=None, bias=None)
    a, b = torch.zeros(1, rows, 3), torch.zeros(1, rows, 3)
    with unittest.mock.patch.object(kb, "causal_conv1d_update", first_kernel):
        try:
            be.forward_extend(layer, batch, torch.zeros(rows, 6), a, b)
        except _Reached:
            return "reached"
        except Exception as e:
            return f"{type(e).__name__}: {e}"
    return "returned before the conv kernel"


def _verify_batch_worker(rank, world, port, backend, kinds, q):
    """DP rank j's forward is kinds[j]. Builds the KDA TP context through the real
    _fast_context (every rank verifies) or _general_context, then runs every target-verify
    stand-in batch through KDAAttnBackend.forward_extend."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        from sglang.srt import runtime_context
        from sglang.srt.layers.communication import k3_kda_tp

        NS = types.SimpleNamespace
        par = _par(
            tp_size=world, attn_dp_size=world, attn_dp_rank=rank, tp_group=_Group(world)
        )
        fb, fm, rows = _local_batch(rank, kinds[rank])
        kb, be = _verify_backend(fm)
        R = BS * D
        out = []
        with (
            unittest.mock.patch.object(k3_kda_tp, "get_parallel", lambda: par),
            unittest.mock.patch.object(k3_kda_tp, "torch", _CpuTorch()),
            unittest.mock.patch.multiple(
                runtime_context,
                get_spec=lambda: NS(speculative_num_draft_tokens=D),
                get_exec=lambda: NS(mamba=NS(enable_mamba_extra_buffer=False)),
            ),
        ):
            if set(kinds) == {"verify"}:
                ctx = k3_kda_tp._fast_context(fb, be, rows, R, {"verify"})
                out.append(("fast", _run_verify(kb, be, ctx.batch, world * R)))
            else:
                ctx = k3_kda_tp._general_context(fb, be, rows, R)
                for src in ctx.sources:
                    if src.batch.forward_mode.is_target_verify():
                        be.forward_metadata = src.metadata
                        be.verify_intermediate_state_indices = (
                            k3_kda_tp.source_scratch_rows(be, src)
                        )
                        verdict = _run_verify(kb, be, src.batch, src.rows)
                        out.append((f"source {src.rank}", verdict))
        q.put((rank, out, 0))
    finally:
        dist.destroy_process_group()


class TestVerifyStandInBatch(unittest.TestCase):
    """KDA TP's stand-in batches must carry what the KDA backend reads from a ForwardBatch
    (target verify reads forward_batch.ragged_verify_layout and spec_info.draft_token_num:
    KDAAttnBackend.forward_extend / _forward_target_verify, kda_backend.py)."""

    maxDiff = None

    def _run(self, kinds, want):
        res, codes = _run_workers(_verify_batch_worker, "gloo", kinds)
        self.assertEqual(codes, [0] * len(kinds))
        for rank, out, _ in res:
            self.assertEqual(out, want, f"rank {rank}")

    def test_fast_path_verify(self):
        self._run(["verify", "verify"], [("fast", "reached")])

    def test_general_path_verify_source(self):
        self._run(["extend", "verify"], [("source 1", "reached")])

    def test_ragged_layout_refused(self):
        from sglang.srt import runtime_context
        from sglang.srt.layers.communication import k3_kda_tp

        fb, fm, rows = _local_batch(0, "verify")
        fb.ragged_verify_layout = object()
        be = types.SimpleNamespace(forward_metadata=fm)
        spec = types.SimpleNamespace(speculative_num_draft_tokens=D)
        with (
            unittest.mock.patch.object(
                k3_kda_tp, "get_parallel", lambda: _par(tp_size=2)
            ),
            unittest.mock.patch.object(runtime_context, "get_spec", lambda: spec),
            self.assertRaisesRegex(NotImplementedError, "ragged verify"),
        ):
            k3_kda_tp._fast_context(fb, be, rows, rows, {"verify"})


def _commit_worker(rank, world, port, backend, counts, q):
    """DP rank j commits counts[j] verify requests (request rows reversed, so the slots come
    from the request table); every rank scatters every rank's requests from its own state
    pool's speculative state."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        from sglang.kernels.ops.mamba import mamba_state_scatter_triton as scatter_mod
        from sglang.srt.layers.communication import k3_kda_tp
        from sglang.srt.mem_cache.memory_pool import MambaPool

        par = _par(
            tp_size=world, attn_dp_size=world, attn_dp_rank=rank, tp_group=_Group(world)
        )
        z = torch.zeros(1)
        mamba_pool = object.__new__(MambaPool)
        mamba_pool.mamba_cache = MambaPool.SpeculativeState(
            conv=[z], temporal=z, intermediate_ssm=z, intermediate_conv_window=[z]
        )
        mapping = torch.arange(CAP, dtype=torch.int32) + rank * CAP + 1
        runner = _runner(
            mamba_pool, req_pool=_req_pool(mamba_pool, size=CAP, mapping=mapping)
        )
        n = counts[rank]
        batch = types.SimpleNamespace(
            req_pool_indices=torch.arange(n).flip(0), mamba_track_indices=None
        )
        last = torch.arange(n) if n else None
        calls = []

        def scatter(caches, slots, last, track, steps, src_indices_raw=None):
            calls.append((caches, slots, last, src_indices_raw))

        with (
            unittest.mock.patch.object(k3_kda_tp, "get_parallel", lambda: par),
            unittest.mock.patch.object(k3_kda_tp, "torch", _CpuTorch()),
            unittest.mock.patch.object(
                scatter_mod, "scatter_mamba_states_after_mtp_verify", scatter
            ),
        ):
            try:
                k3_kda_tp.commit_after_verify(runner, batch, last, None)
            except Exception as e:
                q.put((rank, f"{type(e).__name__}: {e}", 0))
                return
        ((caches, slots, last, rows),) = calls
        out = [
            caches is mamba_pool.mamba_cache,
            slots.tolist(),
            last.tolist(),
            rows.tolist(),
        ]
        q.put((rank, out, 0))
    finally:
        dist.destroy_process_group()


class TestCommitAfterVerify(unittest.TestCase):
    maxDiff = None

    def test_every_rank_commits_every_request(self):
        counts = [2, 0, 3]
        res, codes = _run_workers(_commit_worker, "gloo", counts)
        self.assertEqual(codes, [0] * len(counts))
        slots, last, rows = [], [], []
        for j, n in enumerate(counts):
            slots += [j * CAP + n - i for i in range(n)] + [0] * (CAP - n)
            last += list(range(n)) + [-1] * (CAP - n)
            rows += [j * CAP + i for i in range(CAP)]
        for rank, out, _ in res:
            self.assertEqual(out, [True, slots, last, rows], f"rank {rank}")


class TestRows(unittest.TestCase):
    def test_padded_rows_explicit_counts(self):
        from sglang.srt.layers.communication import k3_kda_tp

        self.assertEqual(k3_kda_tp.padded_rows(2, [1, 3, 0, 2]), 3)
        with self.assertRaisesRegex(RuntimeError, "exceed the DP maximum"):
            k3_kda_tp.padded_rows(4, [1, 3])


if __name__ == "__main__":
    unittest.main()
