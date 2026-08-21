"""FlashInfer backend: attention logit soft capping is a plan-time property.

FlashInfer compiles ``logits_soft_cap`` into the module it selects in ``plan()``
/ ``begin_forward()``. The deprecated ``forward()`` entrypoint still accepts the
argument and then drops it, so a cap that is only handed to ``forward()`` never
reaches the kernel and the model runs uncapped.

The numeric tests below run the backend's own decode/extend paths and compare
against an fp32 reference computed both ways; on a backend that plans without
the cap they match the *uncapped* reference.

    python -m pytest test/registered/unit/layers/attention/test_flashinfer_logits_soft_cap.py -v
"""

import os
import unittest
import unittest.mock
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()

DTYPE = torch.float16
NUM_QO_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 128
LAYER_NUM = 1
POOL_SIZE = 1024
MAX_REQS = 8
CONTEXT_LEN = 512
# Q/K are amplified so the logits land a few multiples of the cap out, where
# tanh() actually reshapes the attention distribution (the regime gemma-2 runs
# in) instead of acting as a no-op.
QK_GAIN = 8.0
LOGIT_CAP = 25.0
DIST_PORT = int(os.environ.get("SGLANG_TEST_DIST_PORT", 29713))

_owned_process_group = False
_prev_server_args = None


def setUpModule():
    """FlashInferAttnBackend reads attn_tp_size off the parallel context."""
    global _owned_process_group, _prev_server_args
    if not _HAS_CUDA or torch.distributed.is_initialized():
        return
    from sglang.srt import runtime_context as rc
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{DIST_PORT}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    _owned_process_group = True

    context = rc.get_context()
    try:
        _prev_server_args = context.server_args
    except ValueError:  # nothing published in this process yet
        _prev_server_args = None
    context.set_server_args(_server_args())


def tearDownModule():
    """Leave no dummy ServerArgs or live TP group behind for later modules."""
    if not _owned_process_group:
        return
    from sglang.srt import runtime_context as rc
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    if _prev_server_args is not None:
        rc.get_context().set_server_args(_prev_server_args)
    destroy_model_parallel()
    destroy_distributed_environment()


def _server_args() -> ServerArgs:
    return ServerArgs(model_path="dummy", disable_cuda_graph=True)


def _build_model(cap: float, capping_method: str = "tanh") -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(
        [
            RadixAttention(
                num_heads=NUM_QO_HEADS,
                head_dim=HEAD_DIM,
                scaling=HEAD_DIM**-0.5,
                num_kv_heads=NUM_KV_HEADS,
                layer_id=i,
                logit_cap=cap,
                logit_capping_method=capping_method,
            )
            for i in range(LAYER_NUM)
        ]
    )
    return model


def _build_runner(model: nn.Module):
    kv_pool = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=DTYPE,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYER_NUM,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    req_to_token_pool = ReqToTokenPool(
        size=MAX_REQS,
        max_context_len=CONTEXT_LEN,
        device="cuda",
        enable_memory_saver=False,
    )

    runner = MagicMock()
    runner.model = model
    runner.req_to_token_pool = req_to_token_pool
    runner.token_to_kv_pool = kv_pool
    runner.token_to_kv_pool_allocator.get_kvcache.return_value = kv_pool
    runner.is_draft_worker = False
    runner.spec_algorithm = SpeculativeAlgorithm.NONE
    runner.device = "cuda"
    runner.dtype = DTYPE
    runner.kv_cache_dtype = DTYPE
    runner.page_size = 1
    runner.sliding_window_size = None
    runner.prefill_attention_backend_str = "flashinfer"
    runner.decode_attention_backend_str = "flashinfer"

    runner.server_args = _server_args()

    model_config = MagicMock()
    model_config.num_attention_heads = NUM_QO_HEADS
    model_config.get_num_kv_heads.return_value = NUM_KV_HEADS
    model_config.get_max_num_attention_heads.return_value = NUM_QO_HEADS
    model_config.head_dim = HEAD_DIM
    model_config.context_len = CONTEXT_LEN
    model_config.is_multimodal = False
    model_config.is_encoder_decoder = False
    model_config.hf_config.architectures = ["Gemma2ForCausalLM"]
    runner.model_config = model_config
    return runner


def _reference_attention(q, k, v, cap: float, causal: bool):
    """fp32 reference: optional tanh soft cap, then softmax."""
    q32 = q.float()
    logits = torch.einsum("qhd,khd->hqk", q32, k.float()) * (HEAD_DIM**-0.5)
    if cap > 0.0:
        logits = cap * torch.tanh(logits / cap)
    if causal:
        q_len, kv_len = logits.shape[-2], logits.shape[-1]
        pos = torch.arange(kv_len, device=logits.device)
        row = torch.arange(q_len, device=logits.device) + (kv_len - q_len)
        logits = logits.masked_fill(pos[None, :] > row[:, None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()


class TestGetModelLogitsSoftCap(CustomTestCase):
    """Unit coverage for the model-level cap lookup itself (no GPU needed)."""

    def _call(self, model):
        from sglang.srt.layers.attention.flashinfer_backend import (
            get_model_logits_soft_cap,
        )

        return get_model_logits_soft_cap(SimpleNamespace(model=model))

    def test_reads_cap_off_radix_attention_layers(self):
        self.assertEqual(self._call(_build_model(LOGIT_CAP)), LOGIT_CAP)

    def test_uncapped_model_reports_zero(self):
        self.assertEqual(self._call(_build_model(0.0)), 0.0)

    def test_model_without_attention_reports_zero(self):
        self.assertEqual(self._call(nn.Linear(4, 4)), 0.0)

    def test_mock_model_runner_reports_zero(self):
        # Unit tests elsewhere build backends off a bare MagicMock runner;
        # .modules() on one is not iterable, so guard rather than raise.
        self.assertEqual(self._call(MagicMock()), 0.0)

    def test_disagreeing_layers_are_rejected(self):
        model = _build_model(LOGIT_CAP)
        model.other = RadixAttention(
            num_heads=NUM_QO_HEADS,
            head_dim=HEAD_DIM,
            scaling=HEAD_DIM**-0.5,
            num_kv_heads=NUM_KV_HEADS,
            layer_id=99,
            logit_cap=LOGIT_CAP + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "layers disagree"):
            self._call(model)

    def test_non_tanh_capping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tanh"):
            self._call(_build_model(LOGIT_CAP, capping_method="clamp"))


@unittest.skipUnless(_HAS_CUDA, "FlashInfer attention backend requires CUDA")
class TestFlashInferLogitsSoftCapKernel(CustomTestCase):
    """The cap has to reach the kernel, not just the ignored forward() kwarg."""

    def _make_backend(self, cap: float):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        model = _build_model(cap)
        runner = _build_runner(model)
        backend = FlashInferAttnBackend(runner)
        return backend, runner, model.layers[0]

    def _decode_once(self, backend, runner, layer, seq_len: int, seed: int):
        from sglang.srt.layers.attention.flashinfer_backend import DecodeMetadata

        torch.manual_seed(seed)
        kv_pool = runner.token_to_kv_pool
        req_to_token = runner.req_to_token_pool.req_to_token

        k_cache = QK_GAIN * torch.randn(
            seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        v_cache = torch.randn(
            seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        # Slot 0 is the reserved cuda-graph padding slot: writes to it are dropped.
        loc = torch.arange(1, seq_len + 1, dtype=torch.int64, device="cuda")
        req_to_token[1, :seq_len] = loc.to(torch.int32)
        kv_pool.set_kv_buffer(layer, loc, k_cache, v_cache)

        q = QK_GAIN * torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        req_pool_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        backend.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            torch.tensor([seq_len], dtype=torch.int32),
            seq_len,
            decode_wrappers=backend.decode_wrappers,
            encoder_lens=None,
            spec_info=None,
        )
        backend.forward_metadata = DecodeMetadata(backend.decode_wrappers)

        forward_batch = SimpleNamespace(
            out_cache_loc=loc[-1:],
            encoder_out_cache_loc=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        )
        out = backend.forward_decode(
            q.view(1, -1), None, None, layer, forward_batch, save_kv_cache=False
        )
        return out.view(1, NUM_QO_HEADS, HEAD_DIM), q, k_cache, v_cache

    def test_decode_output_is_capped(self):
        cap = LOGIT_CAP
        backend, runner, layer = self._make_backend(cap)
        out, q, k, v = self._decode_once(backend, runner, layer, seq_len=64, seed=0)
        ref_capped = _reference_attention(q, k, v, cap, causal=False)
        ref_uncapped = _reference_attention(q, k, v, 0.0, causal=False)

        cos_capped = _cos(out, ref_capped)
        cos_uncapped = _cos(out, ref_uncapped)
        print(
            f"[decode cap={cap}] cos(vs capped)={cos_capped:.6f} "
            f"cos(vs uncapped)={cos_uncapped:.6f}"
        )
        # The two references must be far apart, or the test proves nothing.
        self.assertLess(_cos(ref_capped, ref_uncapped), 0.9)
        self.assertGreater(cos_capped, 0.999)
        self.assertGreater(cos_capped, cos_uncapped)

    def test_extend_output_is_capped(self):
        from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata

        cap = LOGIT_CAP
        backend, runner, layer = self._make_backend(cap)
        seq_len = 32
        torch.manual_seed(1)

        req_to_token = runner.req_to_token_pool.req_to_token
        # Slot 0 is the reserved cuda-graph padding slot: writes to it are dropped.
        loc = torch.arange(1, seq_len + 1, dtype=torch.int64, device="cuda")
        req_to_token[1, :seq_len] = loc.to(torch.int32)

        q = QK_GAIN * torch.randn(
            seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        k = QK_GAIN * torch.randn(
            seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        )
        v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

        req_pool_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([0], dtype=torch.int32, device="cuda")

        backend.indices_updater_prefill.update(
            req_pool_indices,
            seq_lens,
            torch.tensor([seq_len], dtype=torch.int32),
            seq_len,
            prefix_lens,
            prefill_wrappers=backend.prefill_wrappers_paged,
            use_ragged=True,
            encoder_lens=None,
            spec_info=None,
        )
        backend.forward_metadata = PrefillMetadata(
            backend.prefill_wrappers_paged, True, True
        )
        forward_batch = SimpleNamespace(
            out_cache_loc=loc,
            encoder_out_cache_loc=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[seq_len],
        )
        out = backend.forward_extend(
            q.view(seq_len, -1),
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=False,
        ).view(seq_len, NUM_QO_HEADS, HEAD_DIM)

        ref_capped = _reference_attention(q, k, v, cap, causal=True)
        ref_uncapped = _reference_attention(q, k, v, 0.0, causal=True)
        cos_capped = _cos(out, ref_capped)
        cos_uncapped = _cos(out, ref_uncapped)
        print(
            f"[extend cap={cap}] cos(vs capped)={cos_capped:.6f} "
            f"cos(vs uncapped)={cos_uncapped:.6f}"
        )
        self.assertLess(_cos(ref_capped, ref_uncapped), 0.9)
        self.assertGreater(cos_capped, 0.999)
        self.assertGreater(cos_capped, cos_uncapped)

    def test_uncapped_model_matches_uncapped_reference(self):
        """Regression: a model without a cap must be untouched by the change."""
        backend, runner, layer = self._make_backend(0.0)
        out, q, k, v = self._decode_once(backend, runner, layer, seq_len=64, seed=0)
        ref_uncapped = _reference_attention(q, k, v, 0.0, causal=False)
        cos_uncapped = _cos(out, ref_uncapped)
        print(f"[decode cap=0.0] cos(vs uncapped)={cos_uncapped:.6f}")
        self.assertGreater(cos_uncapped, 0.999)


@unittest.skipUnless(_HAS_CUDA, "FlashInfer attention backend requires CUDA")
class TestFlashInferLogitsSoftCapPlanArgs(CustomTestCase):
    """The cap must reach every plan(), including the cuda-graph replay ones."""

    def _plan_kwargs(self, use_fast_decode_plan: bool):
        """Run the decode indices updater and capture what begin_forward got."""
        from sglang.srt.layers.attention import flashinfer_backend as fi

        backend = fi.FlashInferAttnBackend(_build_runner(_build_model(LOGIT_CAP)))
        recorded = {}

        def spy(*args, **kwargs):
            recorded.update(kwargs)

        wrapper = backend.decode_wrappers[0]
        req_pool_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

        with unittest.mock.patch.object(fi, "fast_decode_plan", spy):
            # Capture swaps begin_forward for partial(fast_decode_plan, wrapper);
            # the updater branches on that, so both branches need covering.
            wrapper.begin_forward = partial(spy) if use_fast_decode_plan else spy
            backend.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                torch.tensor([16], dtype=torch.int32),
                16,
                decode_wrappers=backend.decode_wrappers,
                encoder_lens=None,
                spec_info=None,
            )
        return recorded

    def test_decode_plan_receives_cap(self):
        kwargs = self._plan_kwargs(use_fast_decode_plan=False)
        self.assertEqual(kwargs.get("logits_soft_cap"), LOGIT_CAP)

    def test_cuda_graph_decode_plan_receives_cap(self):
        # fast_decode_plan overwrites the wrapper's cap on every replay, so a
        # replay that plans without it silently drops capping inside the graph.
        # The call is spied, so pin the upstream signature it relies on too.
        import inspect

        from sglang.srt.layers.attention.flashinfer_backend import fast_decode_plan

        self.assertIn("logits_soft_cap", inspect.signature(fast_decode_plan).parameters)
        kwargs = self._plan_kwargs(use_fast_decode_plan=True)
        self.assertEqual(kwargs.get("logits_soft_cap"), LOGIT_CAP)

    def test_prefill_plans_receive_cap(self):
        from sglang.srt.layers.attention import flashinfer_backend as fi

        backend = fi.FlashInferAttnBackend(_build_runner(_build_model(LOGIT_CAP)))
        seen = {}

        def spy_for(name):
            def spy(*args, **kwargs):
                seen[name] = kwargs

            return spy

        backend.prefill_wrapper_ragged.begin_forward = spy_for("ragged")
        backend.prefill_wrappers_paged[0].begin_forward = spy_for("paged")
        backend.indices_updater_prefill.update(
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            torch.tensor([16], dtype=torch.int32, device="cuda"),
            torch.tensor([16], dtype=torch.int32),
            16,
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_wrappers=backend.prefill_wrappers_paged,
            use_ragged=True,
            encoder_lens=None,
            spec_info=None,
        )
        self.assertEqual(seen["ragged"].get("logits_soft_cap"), LOGIT_CAP)
        self.assertEqual(seen["paged"].get("logits_soft_cap"), LOGIT_CAP)


class TestFastPrefillPlanCapInvariant(CustomTestCase):
    """fast_prefill_plan reuses the capture-time module; the cap may not move."""

    def test_changed_cap_on_replay_is_rejected(self):
        from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan

        wrapper = SimpleNamespace(
            is_cuda_graph_enabled=True,
            _backend="fa2",
            _cached_module=object(),
            _logits_soft_cap=0.0,
        )
        with self.assertRaisesRegex(AssertionError, "logits_soft_cap changed"):
            fast_prefill_plan(
                wrapper,
                None,
                None,
                None,
                torch.zeros(1, dtype=torch.int32),
                NUM_QO_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                1,
                logits_soft_cap=LOGIT_CAP,
                qo_indptr_host=torch.zeros(2, dtype=torch.int32),
                kv_indptr_host=torch.zeros(2, dtype=torch.int32),
                kv_lens_host=torch.zeros(1, dtype=torch.int32),
                max_q_len=1,
                max_kv_len=1,
            )


if __name__ == "__main__":
    unittest.main()
