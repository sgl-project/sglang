"""One-launch MTP draft-decode replay metadata vs the precompute + copy replay."""

import types
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention import dsa_metadata_bounded as bounded
from sglang.srt.layers.attention.dsa.dsa_draft_metadata import (
    draft_decode_metadata_writable,
    write_draft_decode_metadata,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_bounded_metadata_kit import (
    PAGE,
    TOPK,
    WIDTH_1M,
    MetadataWorld,
    clone_bufs,
    decode_bufs,
    draft_decode_one_launch,
    draft_decode_precompute_copy,
    live_columns,
    table,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PROD = dict(
    program_budget=bounded.DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row=bounded.DEFAULT_MAX_SPLITS_PER_ROW,
)
NOCAP = dict(program_budget=bounded.DEFAULT_PROGRAM_BUDGET, max_splits_per_row=0)
TINY = dict(program_budget=5, max_splits_per_row=256)


def kv_sets(width, seed=1):
    g = torch.Generator().manual_seed(seed)

    def rnd(lo, hi, n):
        return torch.randint(lo, hi, (n,), generator=g).tolist()

    edge = [1, 2, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 8192]
    return {
        "1k": lambda n: rnd(1024, 2048, n),
        "64k": lambda n: rnd(65536, 66560, n),
        "edge": lambda n: [edge[i % len(edge)] for i in range(n)],
        "wide": lambda n: ([width - 1, width + 300] + rnd(1, 3000, n))[:n],
    }


def assert_same(test, expected, actual, tag):
    for name, ref in expected.items():
        got = actual[name]
        if not torch.equal(ref, got):
            diff = (ref != got).nonzero()
            first = tuple(diff[0].tolist())
            test.fail(
                f"{tag}.{name}: {diff.shape[0]} elements differ, first at {first}: "
                f"{ref[first].item()} vs {got[first].item()}"
            )


def assert_same_live(test, expected, actual, seq, width, tag):
    """1-D buffers whole, page tables on each row's live prefix."""
    assert_same(
        test,
        {k: v for k, v in expected.items() if v.dim() == 1},
        actual,
        tag,
    )
    for row, kv_len in enumerate(seq.tolist()):
        cols, real_cols = live_columns(kv_len, width)
        assert_same(
            test,
            {
                "pt1": expected["page_table_1"][row, :cols],
                "real": expected["real_page_table"][row, :real_cols],
            },
            {
                "pt1": actual["page_table_1"][row, :cols],
                "real": actual["real_page_table"][row, :real_cols],
            },
            f"{tag}.row{row}",
        )


class TestOneLaunchKernel(CustomTestCase):
    """fused_dsa_decode_metadata_multi writes, into every destination, what the
    precompute + fused copies write (sentinel-filled scratch, so the copied
    tails are deterministic): 1-5 step backends, padded rows, multi-pass loops."""

    def test_matches_precompute_and_copies(self):
        world = MetadataWorld(WIDTH_1M, pool=40)
        for n_dst in (1, 3, 4, 5):
            for bs in (1, 2, 4, 8, 16):
                for set_name, gen in kv_sets(world.width).items():
                    real = bs - 1 if bs > 2 else bs
                    seq, req = world.batch(gen(bs)[:real], pad_to=bs)
                    for budget in (PROD, NOCAP, TINY):
                        ref = [decode_bufs(world, bs) for _ in range(n_dst)]
                        got = [clone_bufs(b) for b in ref]
                        draft_decode_precompute_copy(world, seq, req, bs, ref)
                        draft_decode_one_launch(world, seq, req, bs, got, **budget)
                        for i in range(n_dst):
                            assert_same(
                                self,
                                ref[i],
                                got[i],
                                f"dst{n_dst}.bs{bs}.{set_name}.{budget}.step{i}",
                            )

    def test_graph_replay_with_new_lengths(self):
        world = MetadataWorld(WIDTH_1M, pool=40, seed=2)
        for bs in (1, 3, 16):
            seq_buf, req_buf = world.batch([1000] * bs)
            dsts = [decode_bufs(world, bs) for _ in range(4)]

            def body():
                draft_decode_one_launch(world, seq_buf, req_buf, bs, dsts)

            body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            for trial, kv in enumerate(
                ([9000] * bs, [3] * bs, [70000] + [129] * (bs - 1))
            ):
                seq, req = world.batch(kv)
                seq_buf.copy_(seq)
                req_buf.copy_(req)
                graph.replay()
                torch.cuda.synchronize()
                ref = [decode_bufs(world, bs) for _ in dsts]
                draft_decode_precompute_copy(world, seq, req, bs, ref)
                for i, dst in enumerate(dsts):
                    assert_same_live(
                        self,
                        ref[i],
                        dst,
                        seq,
                        world.width,
                        f"graph.bs{bs}.t{trial}.{i}",
                    )
            del graph


class _StepBackend:
    """The attributes the one-launch replay reads off a DSA step backend."""

    def __init__(self, world, bs_list, real_schedule=False):
        self.req_to_token = world.req_to_token
        self.dsa_index_topk = TOPK
        self.physical_page_size = PAGE
        self.dsa_meta_program_budget = bounded.DEFAULT_PROGRAM_BUDGET
        self.dsa_meta_max_splits = bounded.DEFAULT_MAX_SPLITS_PER_ROW
        self.dsa_meta_bounded = True
        self.dsa_decode_impl = "trtllm"
        self.forward_metadata = None
        self.prefill_impl_resets = 0
        self.decode_cuda_graph_metadata = {}
        for bs in bs_list:
            dst = decode_bufs(world, bs)
            ctx = sched = None
            if real_schedule:
                ctx = torch.ones((bs, 1), dtype=torch.int32, device="cuda")
                sched = _deepgemm_schedule(ctx).clone()
            self.decode_cuda_graph_metadata[bs] = SimpleNamespace(
                cache_seqlens_int32=dst["cache_seqlens"],
                cu_seqlens_k=dst["cu_seqlens_k"],
                page_table_1=dst["page_table_1"],
                dsa_cache_seqlens_int32=dst["dsa_cache_seqlens"],
                dsa_cu_seqlens_k=dst["dsa_cu_seqlens_k"],
                real_page_table=dst["real_page_table"],
                paged_mqa_schedule_metadata=sched,
                paged_mqa_ctx_lens_2d=ctx,
                topk_v2_plan=None,
                schedule_refreshes=0,
            )
        if real_schedule:
            for name in (
                "_refresh_paged_mqa_schedule_metadata",
                "_refresh_topk_v2_plan",
            ):
                method = getattr(DeepseekSparseAttnBackend, name)
                setattr(self, name, types.MethodType(method, self))

    def _graph_page_table_width(self, metadata):
        return metadata.page_table_1.shape[1]

    def set_dsa_prefill_impl(self, forward_batch=None):
        self.prefill_impl_resets += 1

    def _refresh_paged_mqa_schedule_metadata(self, metadata, seqlens_32_2d):
        metadata.schedule_refreshes += 1

    def _refresh_topk_v2_plan(self, metadata):
        pass


def _deepgemm_schedule(ctx_lens_2d):
    from sglang.srt.layers.attention.dsa.utils import (
        resolve_num_sms,
        resolve_paged_mqa_logits_metadata_fn,
    )

    return resolve_paged_mqa_logits_metadata_fn()(ctx_lens_2d, 64, resolve_num_sms())


def _meta_bufs(metadata):
    return dict(
        cache_seqlens=metadata.cache_seqlens_int32,
        cu_seqlens_k=metadata.cu_seqlens_k,
        page_table_1=metadata.page_table_1,
        dsa_cache_seqlens=metadata.dsa_cache_seqlens_int32,
        dsa_cu_seqlens_k=metadata.dsa_cu_seqlens_k,
        real_page_table=metadata.real_page_table,
    )


def _multi_step(backends, num_steps):
    ms = object.__new__(DeepseekSparseAttnMultiStepBackend)
    ms.speculative_num_steps = num_steps
    ms.attn_backends = backends
    ms._draft_decode_writable = {}
    return ms


class TestOneLaunchReplay(CustomTestCase):
    """The multi-step backend's replay entry takes the one-launch path only when
    every step backend's tables fit the launch, and refreshes the DeepGEMM
    schedule only on draft steps that run the indexer."""

    @classmethod
    def setUpClass(cls):
        cls.world = MetadataWorld(WIDTH_1M, pool=20, seed=4)

    def test_writable_gate(self):
        world, bs = self.world, 3

        def pair():
            backends = [_StepBackend(world, [bs]) for _ in range(2)]
            return backends, [b.decode_cuda_graph_metadata[bs] for b in backends]

        cases = []
        backends, metas = pair()
        cases.append(("ok", backends, True))
        backends, metas = pair()
        wide = table(bs, world.width + PAGE, "cuda")
        metas[1].page_table_1 = wide[:, : world.width]  # same shape, other stride
        cases.append(("stride_mismatch", backends, False))
        backends, metas = pair()
        metas[1].page_table_1 = None
        cases.append(("presence_mismatch", backends, False))
        backends, metas = pair()
        for m in metas:
            m.real_page_table = None
        cases.append(("no_real_page_table", backends, False))
        backends, metas = pair()
        for m in metas:
            m.page_table_1 = None
        cases.append(("real_page_table_only", backends, True))
        backends, metas = pair()
        backends[0].dsa_decode_impl = "flashmla_kv"
        cases.append(("flashmla", backends, False))
        backends, metas = pair()
        backends[0].dsa_meta_bounded = False
        cases.append(("bounded_off", backends, False))
        for name, backends, want in cases:
            with self.subTest(case=name):
                self.assertEqual(draft_decode_metadata_writable(backends, bs), want)
        backends, _ = pair()
        self.assertFalse(draft_decode_metadata_writable(backends, bs + 1))

    def test_replay_entry(self):
        world, bs, num_steps = self.world, 3, 5
        backends = [_StepBackend(world, [bs]) for _ in range(num_steps - 1)]
        ms = _multi_step(backends, num_steps)
        ms.draft_schedule_refresh_steps = ms._draft_schedule_refresh_steps(True)
        seq, req = world.batch([4000, 129, 1], pad_to=bs)
        ref = [decode_bufs(world, bs) for _ in backends]
        draft_decode_precompute_copy(world, seq, req, bs, ref)
        ms.init_forward_metadata_out_graph(
            SimpleNamespace(batch_size=bs, seq_lens=seq, req_pool_indices=req)
        )
        self.assertEqual(ms._draft_decode_writable, {bs: True})
        for i, backend in enumerate(backends):
            metadata = backend.decode_cuda_graph_metadata[bs]
            self.assertIs(backend.forward_metadata, metadata)
            self.assertEqual(backend.prefill_impl_resets, 1)
            self.assertEqual(metadata.schedule_refreshes, 0)
            assert_same_live(
                self, ref[i], _meta_bufs(metadata), seq, world.width, f"step{i}"
            )

    def test_schedule_refresh_steps(self):
        """Seeded draft steps (the draft-extend top-k is reused) skip the
        refresh; unseeded ones refresh the same steps as the copy replay."""
        for num_steps in (2, 3, 4, 5, 6):
            ms = _multi_step([None] * (num_steps - 1), num_steps)
            first = 3 if num_steps > 3 else 0
            self.assertEqual(
                list(ms._draft_schedule_refresh_steps(False)),
                list(range(first, num_steps - 1)),
            )
            self.assertEqual(list(ms._draft_schedule_refresh_steps(True)), [])
        world, bs = self.world, 2
        backends = [_StepBackend(world, [bs]) for _ in range(4)]
        seq, req = world.batch([1500, 9000])
        write_draft_decode_metadata(
            backends=backends,
            seq_lens=seq,
            req_pool_indices=req,
            bs=bs,
            schedule_refresh_steps=range(3, 4),
        )
        self.assertEqual(
            [b.decode_cuda_graph_metadata[bs].schedule_refreshes for b in backends],
            [0, 0, 0, 1],
        )

    def test_unseeded_refresh_uses_deepgemm_schedule(self):
        try:
            _deepgemm_schedule(torch.ones((1, 1), dtype=torch.int32, device="cuda"))
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"paged MQA schedule unavailable: {e!r}")
        world, bs = self.world, 4
        backends = [_StepBackend(world, [bs], real_schedule=True) for _ in range(4)]
        metas = [b.decode_cuda_graph_metadata[bs] for b in backends]
        seq, req = world.batch([1500, 9000, 70, 33000])
        before = [m.paged_mqa_schedule_metadata.clone() for m in metas]
        write_draft_decode_metadata(
            backends=backends,
            seq_lens=seq,
            req_pool_indices=req,
            bs=bs,
            schedule_refresh_steps=range(3, 4),
        )
        ctx = seq.to(torch.int32).view(-1, 1)
        want = _deepgemm_schedule(ctx)
        for i, metadata in enumerate(metas):
            if i == 3:
                self.assertTrue(torch.equal(metadata.paged_mqa_schedule_metadata, want))
                self.assertTrue(torch.equal(metadata.paged_mqa_ctx_lens_2d, ctx))
            else:
                self.assertTrue(
                    torch.equal(metadata.paged_mqa_schedule_metadata, before[i])
                )


if __name__ == "__main__":
    unittest.main()
