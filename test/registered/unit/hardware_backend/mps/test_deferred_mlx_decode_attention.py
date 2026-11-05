"""Small-memory checks for the MLX deferred-commit decode attention kernel."""

from __future__ import annotations

import importlib.util
import unittest

import torch
from packaging.version import Version

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")

_HAS_SUPPORTED_RUNTIME = (
    importlib.util.find_spec("mlx") is not None
    and torch.backends.mps.is_available()
    and Version(torch.__version__) >= Version("2.13.0")
)


def _packed_prefill_inputs(spec, prefix_lengths, extension_lengths, *, pool_slots):
    num_tokens = sum(extension_lengths)
    table_stride = max(prefix_lengths) + 1
    shape = (spec.num_q_heads, spec.head_dim)
    kv_shape = (spec.num_kv_heads, spec.head_dim)
    q = torch.randn(num_tokens, *shape, device="mps", dtype=torch.bfloat16)
    current_k = torch.randn(num_tokens, *kv_shape, device="mps", dtype=torch.bfloat16)
    current_v = torch.randn_like(current_k)
    k_pool = torch.randn(pool_slots, *kv_shape, device="mps", dtype=torch.bfloat16)
    v_pool = torch.randn_like(k_pool)
    slots = torch.randperm(pool_slots, device="mps").to(torch.int32)
    table = torch.zeros(
        len(prefix_lengths), table_stride, device="mps", dtype=torch.int32
    )
    start = 0
    for row, prefix_length in enumerate(prefix_lengths):
        table[row, :prefix_length] = slots[start : start + prefix_length]
        start += prefix_length
    request_rows = torch.arange(len(prefix_lengths), device="mps", dtype=torch.int64)
    return [
        q,
        current_k,
        current_v,
        k_pool,
        v_pool,
        table,
        request_rows,
        torch.tensor(prefix_lengths, device="mps", dtype=torch.int32),
        torch.tensor(extension_lengths, device="mps", dtype=torch.int32),
    ]


def _packed_prefill_reference(inputs, spec):
    """Per-token torch attention; invalid slots are skipped, invalid rows are 0."""
    q, current_k, current_v, k_pool, v_pool, table, request_rows = inputs[:7]
    prefix_lengths, extension_lengths = (t.tolist() for t in inputs[7:])
    q_per_kv = spec.num_q_heads // spec.num_kv_heads
    references = []
    request_start = 0
    for batch, extension_length in enumerate(extension_lengths):
        request = int(request_rows[batch])
        if request < 0:
            references.extend(torch.zeros_like(q[0]) for _ in range(extension_length))
            request_start += extension_length
            continue
        slots = table[request, : prefix_lengths[batch]].long()
        slots = slots[(slots >= 0) & (slots < k_pool.shape[0])]
        for offset in range(extension_length):
            token = request_start + offset
            new_rows = slice(request_start, token + 1)
            keys = torch.cat((k_pool[slots], current_k[new_rows]))
            values = torch.cat((v_pool[slots], current_v[new_rows]))
            keys = keys.repeat_interleave(q_per_kv, dim=1)
            values = values.repeat_interleave(q_per_kv, dim=1)
            scores = torch.einsum("hd,thd->ht", q[token].float(), keys.float())
            probabilities = torch.softmax(scores * spec.attention_scale, dim=-1)
            references.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        request_start += extension_length
    return torch.stack(references)


@unittest.skipUnless(_HAS_SUPPORTED_RUNTIME, "requires Torch 2.13 and MLX on MPS")
class TestMlxDeferredDecodeSmallMemory(unittest.TestCase):
    def test_two_long_sequences_match_torch(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_decode_deferred,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20260811)
        spec = DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128)
        lengths = (128, 129)
        table = torch.arange(258, device="mps", dtype=torch.int32).reshape(2, 129)
        q = torch.randn(2, 16, 128, device="mps", dtype=torch.bfloat16)
        current_k = torch.randn(2, 8, 128, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(258, 8, 128, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)
        request_rows = torch.tensor([0, 1], device="mps", dtype=torch.int64)
        seq_lens = torch.tensor(lengths, device="mps", dtype=torch.int64)

        operation = lambda *arrays: radix_decode_deferred(*arrays, spec=spec)

        def mutation_operation(*arrays):
            output = mx.empty_like(arrays[0])
            output[:] = operation(*arrays)
            return output

        outputs = [
            mlx_call(
                candidate,
                q,
                current_k,
                current_v,
                k_pool,
                v_pool,
                table,
                request_rows,
                seq_lens,
                device="mps",
            )
            for candidate in (
                operation,
                mx.compile(operation, shapeless=False),
                mutation_operation,
                mx.compile(mutation_operation, shapeless=False),
            )
        ]
        references = []
        for row, length in enumerate(lengths):
            slots = table[row, : length - 1].long()
            keys = torch.cat((k_pool[slots], current_k[row : row + 1]))
            values = torch.cat((v_pool[slots], current_v[row : row + 1]))
            keys = keys.repeat_interleave(2, dim=1)
            values = values.repeat_interleave(2, dim=1)
            scores = torch.einsum("hd,thd->ht", q[row].float(), keys.float())
            probabilities = torch.softmax(scores * spec.attention_scale, dim=-1)
            references.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )

        torch.mps.synchronize()
        reference = torch.stack(references).cpu()
        for output in outputs:
            torch.testing.assert_close(output.cpu(), reference, atol=0.008, rtol=0.03)

    def test_generic_shape_matches_torch(self):
        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_decode_deferred,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20260810)
        spec = DeferredAttentionSpec(
            num_q_heads=6,
            num_kv_heads=2,
            head_dim=64,
        )
        q = torch.randn(2, 6, 64, device="mps", dtype=torch.bfloat16)
        current_k = torch.randn(2, 2, 64, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(12, 2, 64, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)
        table = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], device="mps", dtype=torch.int32
        )
        request_rows = torch.tensor([0, 1], device="mps", dtype=torch.int64)
        lengths = torch.tensor([3, 4], device="mps", dtype=torch.int64)

        output = mlx_call(
            lambda *arrays: radix_decode_deferred(*arrays, spec=spec),
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            table,
            request_rows,
            lengths,
            device="mps",
        )
        references = []
        for batch, length in enumerate((3, 4)):
            slots = table[batch, : length - 1].long()
            keys = torch.cat((k_pool[slots], current_k[batch : batch + 1]))
            values = torch.cat((v_pool[slots], current_v[batch : batch + 1]))
            keys = keys.repeat_interleave(3, dim=1)
            values = values.repeat_interleave(3, dim=1)
            scores = torch.einsum("hd,thd->ht", q[batch].float(), keys.float())
            probabilities = torch.softmax(scores * spec.attention_scale, dim=-1)
            references.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )

        torch.mps.synchronize()
        torch.testing.assert_close(
            output.cpu(), torch.stack(references).cpu(), atol=0.006, rtol=0.025
        )

    def test_cold_prefill_causal_gqa_matches_torch(self):
        from sglang.kernels.ops.attention.mlx_radix_attention import (
            DeferredAttentionSpec,
            causal_gqa,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20260802)
        spec = DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128)

        def operation(q, k, v):
            return causal_gqa(q, k, v, spec=spec)

        for token_count in (1, 2, 17):
            with self.subTest(token_count=token_count):
                q = torch.randn(
                    token_count, 16, 128, device="mps", dtype=torch.bfloat16
                )
                k = torch.randn(token_count, 8, 128, device="mps", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                output = mlx_call(operation, q, k, v, device="mps")
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1)[None, ...],
                    k.transpose(0, 1)[None, ...],
                    v.transpose(0, 1)[None, ...],
                    scale=128**-0.5,
                    is_causal=True,
                    enable_gqa=True,
                )[0].transpose(0, 1)
                torch.mps.synchronize()
                torch.testing.assert_close(
                    output.cpu(), reference.cpu(), atol=0.008, rtol=0.03
                )

    def test_packed_prefill_reads_radix_prefix_and_extension_rows(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_prefill_deferred,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20260810)
        spec = DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128)
        extension_lengths = (2, 3)
        extension_lens = torch.tensor(
            extension_lengths, device="mps", dtype=torch.int32
        )
        prefix_lens = torch.tensor([2, 1], device="mps", dtype=torch.int32)
        num_tokens = sum(extension_lengths)
        q = torch.randn(num_tokens, 16, 128, device="mps", dtype=torch.bfloat16)
        current_k = torch.randn(num_tokens, 8, 128, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(10, 8, 128, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)
        table = torch.tensor(
            [[3, 1, 0, 0], [7, 0, 0, 0]], device="mps", dtype=torch.int32
        )
        request_rows = torch.tensor([0, 1], device="mps", dtype=torch.int64)

        compiled_prefill = mx.compile(
            lambda *arrays: radix_prefill_deferred(*arrays, spec=spec)
        )
        output = mlx_call(
            compiled_prefill,
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            table,
            request_rows,
            prefix_lens,
            extension_lens,
            device="mps",
        )

        references = []
        request_start = 0
        for batch_index, extension_len in enumerate(extension_lengths):
            prefix_len = int(prefix_lens[batch_index].cpu())
            cached_slots = table[batch_index, :prefix_len].long()
            cached_k = k_pool[cached_slots]
            cached_v = v_pool[cached_slots]
            for in_request_offset in range(extension_len):
                output_index = request_start + in_request_offset
                keys = torch.cat(
                    (
                        cached_k,
                        current_k[request_start : output_index + 1],
                    )
                ).repeat_interleave(2, dim=1)
                values = torch.cat(
                    (
                        cached_v,
                        current_v[request_start : output_index + 1],
                    )
                ).repeat_interleave(2, dim=1)
                scores = torch.einsum(
                    "hd,thd->ht", q[output_index].float(), keys.float()
                )
                probabilities = torch.softmax(scores * spec.attention_scale, dim=-1)
                references.append(
                    torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                        torch.bfloat16
                    )
                )
            request_start += extension_len

        torch.mps.synchronize()
        torch.testing.assert_close(
            output.cpu(), torch.stack(references).cpu(), atol=0.008, rtol=0.03
        )

    def test_packed_prefill_kernel_paths_match_torch(self):
        """Each case pins one launch geometry the tiled kernel derives from shapes."""
        import mlx.core as mx

        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_prefill_deferred,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        qwen = DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128)
        cases = {
            # Ragged requests: a row-tile crossing (33), an exact two-tile
            # request (64), a short one, no prefix, and a multi-key-tile prefix.
            "ragged_tiles": (qwen, (300, 0, 37), (33, 64, 5)),
            # One row tile, five KV splits over three key tiles: split 2 holds
            # keys rows 0..23 cannot see yet, and splits 3-4 hold no keys.
            "masked_and_empty_splits": (
                DeferredAttentionSpec(num_q_heads=6, num_kv_heads=2, head_dim=64),
                (40,),
                (30,),
            ),
            # head_dim 256 uses the smaller key tile to fit threadgroup memory.
            "head_dim_256": (
                DeferredAttentionSpec(num_q_heads=8, num_kv_heads=2, head_dim=256),
                (40, 5),
                (20, 13),
            ),
            # Two or fewer new tokens per request stay on the per-token kernel.
            "per_token_fallback": (qwen, (10, 20), (1, 1)),
        }
        for name, (spec, prefix_lengths, extension_lengths) in cases.items():
            with self.subTest(name):
                torch.manual_seed(20261104)
                inputs = _packed_prefill_inputs(
                    spec, prefix_lengths, extension_lengths, pool_slots=512
                )
                compiled_prefill = mx.compile(
                    lambda *arrays, spec=spec: radix_prefill_deferred(
                        *arrays, spec=spec
                    )
                )
                output = mlx_call(compiled_prefill, *inputs, device="mps")
                reference = _packed_prefill_reference(inputs, spec)
                torch.mps.synchronize()
                torch.testing.assert_close(
                    output.cpu(), reference.cpu(), atol=0.008, rtol=0.03
                )

    def test_packed_prefill_skips_invalid_slots_and_zeroes_invalid_requests(self):
        """Out-of-pool Radix slots drop out of the softmax; rows of a request
        with an invalid pool index come out as zeros, not uninitialized memory."""
        import mlx.core as mx

        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_prefill_deferred,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call

        torch.manual_seed(20261105)
        spec = DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128)
        inputs = _packed_prefill_inputs(spec, (3, 40, 10), (20, 30, 12), pool_slots=128)
        table, request_rows = inputs[5], inputs[6]
        # Request 0's whole prefix is out of the pool, so its early rows would
        # be dominated by zero keys if invalid slots leaked into the softmax.
        table[0, :3] = torch.tensor([-5, 128 + 7, -1], dtype=torch.int32)
        request_rows[1] = -1
        # Recycle a NaN-filled buffer of the output's size so rows the kernel
        # never writes would read back as NaN instead of a lucky zero.
        mx.eval(mx.full(tuple(inputs[0].shape), mx.nan, dtype=mx.bfloat16))

        output = mlx_call(
            lambda *arrays: radix_prefill_deferred(*arrays, spec=spec),
            *inputs,
            device="mps",
        )
        reference = _packed_prefill_reference(inputs, spec)
        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0.008, rtol=0.03)
        torch.testing.assert_close(
            output[20:50].cpu(), torch.zeros_like(output[20:50]).cpu(), atol=0, rtol=0
        )

    def test_current_token_bypasses_uncommitted_pool_entry(self):
        from sglang.kernels.ops.attention._deferred_radix_attention_mlx import (
            DeferredAttentionSpec,
            radix_decode_deferred,
        )
        from sglang.srt.utils.tensor_bridge import (
            borrow_torch_tensors,
            mlx_to_torch,
        )

        torch.manual_seed(20260801)
        batch_size = 4
        pool_slots = 31
        table_stride = 8
        # Every adjacent pair is owned by one KV head.  Make the sibling
        # queries deliberately different so a GQA kernel cannot accidentally
        # reuse the first query's online-softmax state for the second.
        q_per_kv_head = torch.randn(
            batch_size, 8, 128, device="mps", dtype=torch.bfloat16
        )
        q = torch.stack((q_per_kv_head, -q_per_kv_head), dim=2).reshape(
            batch_size, 16, 128
        )
        current_k = torch.randn(batch_size, 8, 128, device="mps", dtype=torch.bfloat16)
        current_v = torch.randn_like(current_k)
        k_pool = torch.randn(pool_slots, 8, 128, device="mps", dtype=torch.bfloat16)
        v_pool = torch.randn_like(k_pool)
        req_to_token = torch.zeros(3, table_stride, device="mps", dtype=torch.int32)
        req_to_token[1] = torch.tensor(
            [-17, 21, 7, 18, 2, 29, 11, 5], device="mps", dtype=torch.int32
        )
        req_to_token[2] = torch.tensor(
            [14, 4, 23, 1, 27, 8, 38, 6], device="mps", dtype=torch.int32
        )
        req_pool_indices = torch.tensor([1, 2, -1, 0], device="mps", dtype=torch.int64)
        seq_lens = torch.tensor([1, 7, 4, 0], device="mps", dtype=torch.int64)

        original_k_pool = k_pool.clone()
        original_v_pool = v_pool.clone()
        torch.mps.synchronize()
        views = borrow_torch_tensors(
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
            synchronize=False,
        )
        result = radix_decode_deferred(
            *(view.array for view in views),
            spec=DeferredAttentionSpec(num_q_heads=16, num_kv_heads=8, head_dim=128),
        )
        output = mlx_to_torch(result, device="mps")

        references = []
        scale = 128**-0.5
        for batch_index, sequence_length in enumerate((1, 7)):
            request = int(req_pool_indices[batch_index].item())
            slots = req_to_token[request, : sequence_length - 1].long()
            keys = torch.cat(
                (k_pool[slots], current_k[batch_index : batch_index + 1]), dim=0
            )
            values = torch.cat(
                (v_pool[slots], current_v[batch_index : batch_index + 1]), dim=0
            )
            keys = keys.repeat_interleave(2, dim=1)
            values = values.repeat_interleave(2, dim=1)
            logits = (
                torch.einsum("hd,thd->ht", q[batch_index].float(), keys.float()) * scale
            )
            probabilities = torch.softmax(logits, dim=-1)
            references.append(
                torch.einsum("ht,thd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        # Invalid request metadata and a non-positive sequence length are
        # threadgroup-uniform early exits.  Both query heads paired with every
        # KV head must be initialized, not merely the first sibling.
        references.extend(
            torch.zeros(16, 128, device="mps", dtype=torch.bfloat16) for _ in range(2)
        )
        reference = torch.stack(references)

        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0.004, rtol=0.02)
        torch.testing.assert_close(
            output[2:].cpu(),
            torch.zeros_like(output[2:]).cpu(),
            atol=0.0,
            rtol=0.0,
        )
        # The final logical entries deliberately contain invalid/stale pool
        # indices.  The kernel must use current K/V instead of reading those
        # entries, including the no-prefix sequence_length == 1 case.
        torch.testing.assert_close(k_pool.cpu(), original_k_pool.cpu())
        torch.testing.assert_close(v_pool.cpu(), original_v_pool.cpu())


if __name__ == "__main__":
    unittest.main()
