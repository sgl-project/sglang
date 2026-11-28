"""Checks that the FlashInfer GDN prefill prewarm leaves no kernel to compile."""

import contextlib
import functools
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_exec
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    build_gdn_attention_fixture,
    run_gdn_fixture_eager,
)

# FlashInfer's DSL prefill kernels require head size 128 on SM90 and SM100.
PREWARM_HEAD_DIM = 128


def prewarm_case(
    name, num_k_heads, num_v_heads, prefix_lens, extend_lens, track_seqlens
):
    """A FlashInfer prefill batch, and the track lengths of its checkpointed run."""
    return (
        GDNAttentionCase(
            name=name,
            backend="triton",
            linear_attn_prefill_backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            page_size=16,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
        ),
        track_seqlens,
    )


def assert_prewarm_leaves_nothing_to_compile(test, cases):
    """FlashInfer compiles each GDN prefill variant on its first call, which
    stalls serving for seconds; after the startup prewarm, no prefill of the
    given batches, with or without tracked checkpoints, compiles."""
    from flashinfer.gdn_kernels.delta_rule_dsl import custom_compile_cache

    fixtures = [
        build_gdn_attention_fixture(
            test,
            case,
            head_k_dim=PREWARM_HEAD_DIM,
            head_v_dim=PREWARM_HEAD_DIM,
            max_context_len=320,
            # Room for each request's slot and its tracked checkpoint's.
            runner_batch_size=10,
        )
        for case, _ in cases
    ]
    with contextlib.ExitStack() as stack:
        # Start from empty compile caches, as a new server process does.
        stack.enter_context(
            patch.dict(custom_compile_cache._in_mem_compile_cache, clear=True)
        )
        compiled = [lambda: len(custom_compile_cache._in_mem_compile_cache)]
        if torch.cuda.get_device_capability()[0] == 10:
            from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill, gdn_prefill

            fresh = functools.cache(gdn_prefill._get_compiled_cache.__wrapped__)
            stack.enter_context(patch.object(gdn_prefill, "_get_compiled_cache", fresh))
            compiled.append(lambda: fresh.cache_info().currsize)
            # The context-parallel fixup kernels, one per specialization.
            fixups = functools.cache(gdn_cp_prefill._get_fixup_kernel.__wrapped__)
            stack.enter_context(
                patch.object(gdn_cp_prefill, "_get_fixup_kernel", fixups)
            )
            compiled.append(lambda: fixups.cache_info().currsize)
        stack.enter_context(get_exec().mamba.override(enable_mamba_extra_buffer=True))

        linear_backend = fixtures[0].backend.linear_attn_backend
        linear_backend.prewarm_flashinfer_extend(fixtures[0].actual_module)
        prewarmed = [count() for count in compiled]
        test.assertGreater(sum(prewarmed), 0)

        for fixture, (case, track_seqlens) in zip(fixtures, cases):
            run_gdn_fixture_eager(fixture)
            test.assertEqual([count() for count in compiled], prewarmed, case.name)
            batch = fixture.forward_batch
            batch.mamba_track_mask = torch.ones(
                case.batch_size, dtype=torch.bool, device="cuda"
            )
            batch.mamba_track_indices = torch.arange(
                case.batch_size + 1,
                2 * case.batch_size + 1,
                dtype=torch.int64,
                device="cuda",
            )
            batch.mamba_track_seqlens = torch.tensor(
                track_seqlens, dtype=torch.int64, device="cuda"
            )
            run_gdn_fixture_eager(fixture)
            metadata = fixture.backend.linear_attn_backend.forward_metadata
            test.assertGreater(metadata.num_state_checkpoints, 0)
            test.assertEqual([count() for count in compiled], prewarmed, case.name)
