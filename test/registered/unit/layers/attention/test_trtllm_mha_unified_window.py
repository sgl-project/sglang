"""TRT-LLM MHA attention over a unified pool with one window sub-pool: the
window layer reads through its window sub-pool's page table, extend and
decode, and matches attention computed from positions with the layer's
window.

Runs where the backend dispatches to the trtllm-gen kernels (SM100)."""

import unittest

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.utils import is_flashinfer_available, is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods import (
    dense_attention as _kit,
)
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    MockModelRunner,
    TinyModelConfig,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

# The kit enters a process-wide single-rank parallel override when imported;
# keep it to this module's tests so later tests in the process see none.
_kit._parallel_override.__exit__(None, None, None)

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")

_HEADS, _KV_HEADS, _HEAD_DIM = 4, 2, 64
_DTYPE = torch.bfloat16
_PAGE = 16
_CONTEXT = 128
_WINDOW = 32
_ATOL = _RTOL = 3e-2


def _storage():
    geometry = MHAGeometry(
        head_num=_KV_HEADS,
        head_dim=_HEAD_DIM,
        v_head_dim=_HEAD_DIM,
        kv_cache_dtype=_DTYPE,
    )
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=geometry,
        windows=[WindowLayers(layer_ids=(1,), geometry=geometry, window=_WINDOW)],
    )
    plan = plan_kv_storage(decls, page_size=_PAGE)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=2048,
            window_tokens=2048,
            state_slots=0,
            max_running_requests=4,
            context_len=_CONTEXT,
            alignment=_PAGE,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cuda",
        start_layer=0,
        end_layer=2,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=4,
        model_context_len=_CONTEXT,
        lazy_compaction=True,
    )
    return build_kv_storage(plan, sized, ctx)


class _Model:
    """A unified pool over a full layer and a window layer, a TRT-LLM MHA
    backend bound to its runner, and the K/V each layer has written."""

    def __init__(self):
        setup_allocator_context()
        storage = _storage()
        case = DenseAttentionCase(
            name="trtllm_mha_unified_window",
            backend="trtllm_mha",
            forward_mode=ForwardMode.EXTEND,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            page_size=_PAGE,
            prefix_lens=(0,) * 4,
            sliding_window_size=_WINDOW,
        )
        runner = MockModelRunner(
            case=case,
            model_config=TinyModelConfig(
                num_heads=_HEADS,
                num_kv_heads=_KV_HEADS,
                head_dim=_HEAD_DIM,
                hidden_size=_HEADS * _HEAD_DIM,
                context_len=_CONTEXT,
                sliding_window_size=_WINDOW,
            ),
            dtype=_DTYPE,
            device="cuda",
            max_context_len=_CONTEXT,
            head_dim=_HEAD_DIM,
            runner_batch_size=4,
        )
        self.allocator = storage.allocator
        runner.req_to_token_pool = storage.requests
        runner.token_to_kv_pool = self.allocator.get_kvcache()
        runner.token_to_kv_pool_allocator = self.allocator
        runner.sliding_window_size = _WINDOW
        runner.init_runner_ids()
        self.runner = runner
        self.backend = TRTLLMHAAttnBackend(runner)
        runner.runner_ids.bind_and_verify_backends([self.backend])
        self.layers = [
            RadixAttention(
                num_heads=_HEADS,
                head_dim=_HEAD_DIM,
                scaling=_HEAD_DIM**-0.5,
                num_kv_heads=_KV_HEADS,
                layer_id=layer_id,
                sliding_window_size=window,
            )
            for layer_id, window in enumerate((-1, _WINDOW))
        ]
        self.ids = {}
        self.kv = {}

    def add_request(self, row, length, generator):
        pages = -(-length // _PAGE)
        self.ids[row] = self.allocator.alloc(pages * _PAGE)[:length]
        self.runner.req_to_token_pool.req_to_token[row, :length] = self.ids[row].to(
            torch.int32
        )
        self.kv[row] = [
            tuple(
                torch.randn(
                    (length, _KV_HEADS, _HEAD_DIM), generator=generator, device="cuda"
                ).to(_DTYPE)
                for _ in range(2)
            )
            for _ in self.layers
        ]

    def batch(self, mode, rows, prefix_lens, input_lens):
        seq_lens = [p + n for p, n in zip(prefix_lens, input_lens)]
        positions = [p + i for p, n in zip(prefix_lens, input_lens) for i in range(n)]
        out = torch.cat(
            [self.ids[r][p : p + n] for r, p, n in zip(rows, prefix_lens, input_lens)]
        )
        batch = ForwardBatch(
            forward_mode=mode,
            batch_size=len(rows),
            input_ids=torch.zeros(len(positions), dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.tensor(rows, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            out_cache_loc=out,
            seq_lens_sum=sum(seq_lens),
            positions=torch.tensor(positions, dtype=torch.int64, device="cuda"),
        )
        self.runner.runner_ids.bind_own_plan(batch)
        if mode.is_extend():
            batch.extend_prefix_lens = torch.tensor(
                prefix_lens, dtype=torch.int32, device="cuda"
            )
            batch.extend_prefix_lens_cpu = list(prefix_lens)
            batch.extend_seq_lens = torch.tensor(
                input_lens, dtype=torch.int32, device="cuda"
            )
            batch.extend_seq_lens_cpu = list(input_lens)
            batch.extend_num_tokens = sum(input_lens)
        return batch

    def inputs(self, rows, prefix_lens, input_lens, generator):
        n = sum(input_lens)
        result = []
        for layer_id in range(len(self.layers)):
            q = torch.randn(
                (n, _HEADS * _HEAD_DIM), generator=generator, device="cuda"
            ).to(_DTYPE)
            k, v = (
                torch.cat(
                    [
                        self.kv[r][layer_id][i][p : p + m]
                        for r, p, m in zip(rows, prefix_lens, input_lens)
                    ]
                )
                for i in range(2)
            )
            result.append((q, k, v))
        return result

    def forward(self, batch, inputs):
        fn = (
            self.backend.forward_decode
            if batch.forward_mode.is_decode()
            else self.backend.forward_extend
        )
        return [
            fn(q, k, v, layer, batch) for layer, (q, k, v) in zip(self.layers, inputs)
        ]

    def reference(self, mode, rows, prefix_lens, input_lens, inputs):
        """A window layer's query at position p reads keys from p - window
        through p (the kernels' ``window_left``), extend and decode alike."""
        outputs = []
        for layer, (q, _, _) in zip(self.layers, inputs):
            window = layer.sliding_window_size
            out, start = [], 0
            for r, p, m in zip(rows, prefix_lens, input_lens):
                keys, values = self.kv[r][layer.layer_id]
                for i in range(m):
                    pos = p + i
                    first = 0 if window < 0 else max(0, pos - window)
                    query = q[start + i].view(_HEADS, _HEAD_DIM).float()
                    group = _HEADS // _KV_HEADS
                    k_ = keys[first : pos + 1].float().repeat_interleave(group, dim=1)
                    v_ = values[first : pos + 1].float().repeat_interleave(group, dim=1)
                    scores = torch.einsum("hd,khd->hk", query, k_) * layer.scaling
                    probs = torch.softmax(scores, dim=-1)
                    out.append(torch.einsum("hk,khd->hd", probs, v_).reshape(-1))
                start += m
            outputs.append(torch.stack(out).to(_DTYPE))
        return outputs


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available() and is_sm100_supported(),
    "the trtllm-gen kernels need SM100",
)
class TestTRTLLMMHAUnifiedWindow(CustomTestCase):
    def setUp(self):
        override = get_parallel().override(attn_tp_size=1, attn_dcp_rank=0)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def tearDown(self):
        # The pools published a runtime context; later tests start from none.
        reset_context()

    def _check(self, actual, expected, tag):
        for layer_id, (a, e) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                a.view_as(e),
                e,
                atol=_ATOL,
                rtol=_RTOL,
                msg=lambda m: f"{tag} layer {layer_id}: {m}",
            )

    def test_window_layer_reads_its_window_pages(self):
        model = _Model()
        generator = torch.Generator(device="cuda").manual_seed(3)
        rows = [1, 2, 3]
        prefix_lens, extend_lens = [40, 70, 9], [5, 9, 30]
        for row, p, n in zip(rows, prefix_lens, extend_lens):
            model.add_request(row, p + n + 1, generator)
        zeros = [0] * len(rows)
        for prefix, lens, tag in (
            (zeros, prefix_lens, "prefix"),
            (prefix_lens, extend_lens, "extend"),
        ):
            batch = model.batch(ForwardMode.EXTEND, rows, prefix, lens)
            inputs = model.inputs(rows, prefix, lens, generator)
            model.backend.init_forward_metadata(batch)
            actual = model.forward(batch, inputs)
            expected = model.reference(ForwardMode.EXTEND, rows, prefix, lens, inputs)
            self._check(actual, expected, tag)
        seq = [p + n for p, n in zip(prefix_lens, extend_lens)]
        ones = [1] * len(rows)
        batch = model.batch(ForwardMode.DECODE, rows, seq, ones)
        inputs = model.inputs(rows, seq, ones, generator)
        model.backend.init_forward_metadata(batch)
        actual = model.forward(batch, inputs)
        expected = model.reference(ForwardMode.DECODE, rows, seq, ones, inputs)
        self._check(actual, expected, "decode")


if __name__ == "__main__":
    unittest.main()
