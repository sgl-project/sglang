"""SM70 CUDA CSA2 vs the torch oracle at long context (no Hopper).

Prefill-vs-decode at T=2048 (serving chunk) plus last-row indexer at 8k and
30k. Skips when CUDA is missing or the device is not SM70. Do not run this
against a live TP8 serve job.

This is two last-row scores plus one chunked prefill (~10 s). It is not
30 min of TG. Live decode soak:
test/manual/dsv41_v100/test_tg_soak_cohesion.py
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.srt.layers.attention.dsv4.sm70_csa2 import (
    get_state,
    sm70_forward_low_ratio_sources,
    sm70_forward_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

FP16 = torch.float16
LOGIT_RTOL = 2**-10
ATOL = 2e-2
RTOL = 2e-2
LAYER_ORDER = (2, 20, 24)
FLASH_TOPK = 512


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")
    venv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isdir(venv_bin):
        path = os.environ.get("PATH", "")
        if venv_bin not in path.split(os.pathsep):
            os.environ["PATH"] = venv_bin + os.pathsep + path
    if shutil.which("ninja") is None:
        raise unittest.SkipTest("ninja required to JIT sm70_dsv41_csa2")
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * (1 << 30):
        raise unittest.SkipTest(
            f"need >=4 GiB free GPU memory, have {free / (1 << 30):.1f} GiB"
        )


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = set(a[a >= 0].tolist())
    sb = set(b[b >= 0].tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _lin(v, weight):
    return (v.float() @ weight.float().t()).to(v.dtype)


class TestSm70CudaLongctx(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_sm70()
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        torch.set_num_threads(4)
        cls.dev = torch.device("cuda")
        from sglang.kernels.ops.attention import sm70_dsv41_csa2 as csa2

        cls.csa2 = csa2

    def test_index_logits_last_row_vs_oracle_8k_and_30k(self):
        g = torch.Generator(device="cpu").manual_seed(2)
        h, d, ratio = 8, 128, 1
        print("\n--- CUDA index_logits last-row vs torch oracle ---")
        print(f"{'n':>6} {'maxabs':>10} {'rtol*scale':>12}")
        for n in (8192, 30000):
            q = torch.randn(1, h, d, generator=g, dtype=torch.float32).to(FP16)
            k = torch.randn(n, d, generator=g, dtype=torch.float32).to(FP16)
            w = torch.randn(1, h, generator=g, dtype=torch.float32).to(FP16)
            query_pos = torch.tensor([n - 1], dtype=torch.int32)
            kq = R.fake_quant_fp4_ue8m0(k)
            pay, exp = R.pack_index_fp4_ue8m0(k)
            rows = torch.cat([pay, exp], dim=-1)
            ref = R.index_scores(q, kq, w, torch.float32)
            vis = (query_pos + 1) // ratio
            reach = torch.arange(n)[None, :] < vis[:, None]
            ref = ref.masked_fill(~reach, -torch.inf)
            got = self.csa2.index_logits(
                q.to(self.dev),
                w.to(self.dev),
                rows.to(self.dev),
                query_pos.to(self.dev),
                ratio,
            ).cpu()
            finite = torch.isfinite(ref)
            self.assertTrue(torch.equal(torch.isfinite(got), finite), f"n={n} finite mask")
            scale = ref[finite].abs().amax().clamp_min(1e-4)
            err = (got[finite] - ref[finite]).abs().amax().item()
            lim = LOGIT_RTOL * float(scale)
            print(f"{n:6d} {err:10.4e} {lim:12.4e}")
            self.assertLessEqual(err, lim, f"n={n} index_logits vs oracle")
            idx_ref = R.topk_positions(ref, torch.tensor([[n]]), FLASH_TOPK)[0]
            idx_got = R.topk_positions(got, torch.tensor([[n]]), FLASH_TOPK)[0]
            jac = _jaccard(idx_ref, idx_got)
            print(f"       topk Jaccard={jac:.4f}")
            self.assertGreaterEqual(jac, 0.95, f"n={n} last-row top-512 Jaccard")

    def test_prefill_vs_decode_at_chunk_2048(self):
        """Same check as T=700 in the short oracle, at the serving prefill chunk."""
        t = 2048
        extra = 3
        need = t + extra
        g = torch.Generator(device="cpu").manual_seed(7)
        hidden, heads, q_lora_r, n_ih = 64, 4, 32, 8
        dev = self.dev
        freqs = R.precompute_freqs_cis(64, need + 8, 0, 10000.0, 1.0, 32.0, 1.0)[
            :need
        ].to(dev)
        wkv = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(dev)
        kv_norm = torch.ones(512, dtype=FP16, device=dev)
        wq_b = (torch.randn(n_ih * 128, q_lora_r, generator=g).to(FP16) * 0.02).to(dev)
        w_proj = (torch.randn(n_ih, hidden, generator=g).to(FP16) * 0.02).to(dev)
        wk = (torch.randn(128, 512, generator=g).to(FP16) * 0.02).to(dev)
        k_norm = torch.ones(128, dtype=FP16, device=dev)
        wkv_c = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(dev)
        wgate = (torch.randn(512, hidden, generator=g).to(FP16) * 0.02).to(dev)
        norm_c = torch.ones(512, dtype=FP16, device=dev)
        sink = torch.zeros(heads, dtype=torch.float32, device=dev)
        x = (torch.randn(need, hidden, generator=g).to(FP16) * 0.05).to(dev)
        q_lora = (torch.randn(need, q_lora_r, generator=g).to(FP16) * 0.05).to(dev)
        q = (torch.randn(need, heads, 512, generator=g).to(FP16) * 0.1).to(dev)
        positions = torch.arange(need, dtype=torch.int64, device=dev)

        def indexer(owns_k, is_src, uses_cand):
            return SimpleNamespace(
                owns_k=owns_k,
                is_candidate_source=is_src,
                uses_candidates=uses_cand,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
                index_topk=512,
                wk=SimpleNamespace(weight=wk),
                k_norm=SimpleNamespace(weight=k_norm),
                queries=lambda ql, fr: R.fake_quant_fp4_ue8m0(
                    R.rope_tail(_lin(ql, wq_b).view(ql.shape[0], 8, 128), fr, 64)
                ),
                head_weights=lambda xx: _lin(xx, w_proj) * (128**-0.5 * 8**-0.5),
            )

        def layer(lid, ratio, compressor, idx):
            return SimpleNamespace(
                layer_id=lid,
                compress_ratio=ratio,
                head_dim=512,
                qk_rope_head_dim=64,
                eps=1e-20,
                sliding_window=128,
                softmax_scale=512**-0.5,
                freqs_cis=freqs,
                wkv=lambda v: (_lin(v, wkv), None),
                kv_norm=SimpleNamespace(weight=kv_norm),
                compressor=compressor,
                indexer=idx,
                attn_sink=sink,
            )

        c1 = SimpleNamespace(
            wkv=SimpleNamespace(weight=wkv_c),
            norm=SimpleNamespace(weight=norm_c),
        )
        c2 = SimpleNamespace(
            wkv=SimpleNamespace(weight=wkv_c),
            wgate=SimpleNamespace(weight=wgate),
            norm=SimpleNamespace(weight=norm_c),
        )
        layers = {
            2: layer(2, 2, c2, indexer(True, False, False)),
            20: layer(20, 1, c1, indexer(True, True, False)),
            24: layer(24, 1, None, indexer(False, False, True)),
        }

        def fb(decode: bool):
            return SimpleNamespace(
                batch_size=1,
                forward_mode=SimpleNamespace(
                    is_decode=lambda: decode,
                    is_extend=lambda: not decode,
                    is_target_verify=lambda: False,
                ),
            )

        def run_extend(backend, ly, sl):
            mode = fb(False)
            outs = {}
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, ly[lid], x[sl], q_lora[sl], positions[sl], mode
                )
                outs[lid] = sm70_forward_sparse(
                    backend, q[sl], ly[lid], mode, ly[lid].compress_ratio, sink
                )
            return outs

        def run_decode(backend, ly, i):
            mode = fb(True)
            sl = slice(i, i + 1)
            outs = {}
            for lid in LAYER_ORDER:
                sm70_forward_low_ratio_sources(
                    backend, ly[lid], x[sl], q_lora[sl], positions[sl], mode
                )
                outs[lid] = sm70_forward_sparse(
                    backend, q[sl], ly[lid], mode, ly[lid].compress_ratio, sink
                )
            return outs

        be_a = SimpleNamespace(model_runner=None, max_context_len=need + 8)
        outs_a = run_extend(be_a, layers, slice(0, t))
        be_b = SimpleNamespace(model_runner=None, max_context_len=need + 8)
        layers_b = {
            2: layer(2, 2, c2, indexer(True, False, False)),
            20: layer(20, 1, c1, indexer(True, True, False)),
            24: layer(24, 1, None, indexer(False, False, True)),
        }
        run_extend(be_b, layers_b, slice(0, t - 1))
        outs_b = run_decode(be_b, layers_b, t - 1)
        print("\n--- CUDA prefill vs decode T=2048 ---")
        for lid in LAYER_ORDER:
            d = float((outs_a[lid][-1:].float() - outs_b[lid].float()).abs().max())
            print(f"L{lid} maxabs={d:.5f}")
            self.assertTrue(
                torch.allclose(
                    outs_a[lid][-1:].float(), outs_b[lid].float(), atol=ATOL, rtol=RTOL
                ),
                f"T=2048 L{lid} maxabs={d}",
            )
        st_a, st_b = get_state(be_a), get_state(be_b)
        for lid, key in ((20, "topk"), (24, "topk")):
            jac = _jaccard(st_b.topk[lid][0], st_a.prefill_topk[lid][-1])
            print(f"L{lid} topk Jaccard={jac:.4f}")
            self.assertGreaterEqual(jac, 0.95, f"L{lid} topk Jaccard {jac}")
        jac_c = _jaccard(st_b.cand_ids[0], st_a.prefill_cand_ids[-1])
        print(f"L20 cand Jaccard={jac_c:.4f}")
        self.assertGreaterEqual(jac_c, 0.95)


if __name__ == "__main__":
    unittest.main()
