"""CPU checks of the megakernel binding: the view is the running forward's attention
backend and KV pool, and a source replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def set_swa_key_buffer_radix_fused_norm_rope(self, **kw):
        return kw


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata="core"),
        trtllm_attn=False,
        forward=lambda *a, **k: "attention",
        store_cache=lambda **k: None,
        get_swa_out_cache_loc=lambda fb: "swa_loc",
        forward_low_ratio_sources=lambda **k: "sources",
        forward_core_compressor=lambda **k: "compressed",
    )


class FullPool:
    page_size = 2
    dtype = torch.float8_e4m3fn
    store_dtype = torch.uint8

    def __init__(self):
        self.size = 8
        self.kv_buffer = [
            torch.zeros(10, 1, 3, dtype=self.store_dtype) for _ in range(2)
        ]


class TokenPool(HybridLinearKVPool):
    use_mla = True
    post_capture_active = True

    def __init__(self):
        self.size = 8
        self.full_kv_pool = FullPool()
        self.full_attention_layer_id_mapping = {1: 0, 3: 1}
        self.key_buffer_calls = []

    def get_key_buffer(self, layer_id):
        self.key_buffer_calls.append(layer_id)
        raw = self.full_kv_pool.kv_buffer[
            self.full_attention_layer_id_mapping[layer_id]
        ]
        return raw.view(self.full_kv_pool.dtype)


class MambaPool:
    def __init__(self):
        self.size = 5
        self.conv = [torch.zeros(2, self.size, 3)]
        self.temporal = torch.zeros(2, self.size, 4)


class RequestPool(HybridReqToTokenPool):
    def __init__(self):
        self.mamba_pool = MambaPool()
        self.mamba_map = {0: 0, 2: 1}

    def mamba2_layer_cache(self, layer_id):
        dense = self.mamba_map[layer_id]
        return SimpleNamespace(
            conv=[self.mamba_pool.conv[0][dense]],
            temporal=self.mamba_pool.temporal[dense],
            replayssm_d=None,
        )


def hybrid_backend():
    token_pool, request_pool = TokenPool(), RequestPool()
    full_metadata = SimpleNamespace(
        block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens_k=torch.zeros(2, dtype=torch.int32),
    )
    tracked = []

    def child(cls, **kw):
        return Mock(
            spec=cls,
            token_to_kv_pool=token_pool,
            req_to_token_pool=request_pool,
            **kw,
        )

    full = child(TRTLLMMLABackend, forward_decode_metadata=full_metadata)

    def track(*args):
        tracked.append(args)

    linear = child(
        KDAAttnBackend,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.zeros(2, dtype=torch.int32)
        ),
        _track_mamba_state_decode=track,
    )
    value = Mock(spec=HybridLinearAttnBackend, token_to_kv_pool=token_pool)
    value.req_to_token_pool = request_pool
    value.full_attn_backend = full
    value.linear_attn_backend = linear
    value.attn_backend_list = [full, linear]
    runner = SimpleNamespace(attn_backend=value)
    return SimpleNamespace(
        runner=runner,
        backend=value,
        full=full,
        linear=linear,
        token_pool=token_pool,
        full_pool=token_pool.full_kv_pool,
        request_pool=request_pool,
        mamba_pool=request_pool.mamba_pool,
        tracked=tracked,
    )


def forward_batch():
    return SimpleNamespace(
        batch_size=2,
        forward_mode="decode",
        out_cache_loc=torch.zeros(2, dtype=torch.int32),
        out_cache_loc_is_physical=True,
    )


class TestMegakernelBinding(CustomTestCase):
    def test_hybrid_preparation_requires_the_supported_backend_and_pool_types(self):
        cases = {
            "lookalike": lambda f: setattr(
                f.runner,
                "attn_backend",
                SimpleNamespace(
                    full_attn_backend=f.full,
                    linear_attn_backend=f.linear,
                    attn_backend_list=[f.full, f.linear],
                    token_to_kv_pool=f.token_pool,
                    req_to_token_pool=f.request_pool,
                ),
            ),
            "other_full_backend": lambda f: setattr(
                f.backend, "full_attn_backend", SimpleNamespace()
            ),
            "other_linear_backend": lambda f: setattr(
                f.backend, "linear_attn_backend", SimpleNamespace()
            ),
            "other_token_pool": lambda f: setattr(
                f.backend, "token_to_kv_pool", SimpleNamespace()
            ),
            "other_request_pool": lambda f: setattr(
                f.backend, "req_to_token_pool", SimpleNamespace()
            ),
            "non_mla_pool": lambda f: setattr(f.token_pool, "use_mla", False),
        }
        for name, replace in cases.items():
            with self.subTest(name=name):
                fixture = hybrid_backend()
                replace(fixture)
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                self.assertIsNone(binding.preparation_view(fixture.runner).hybrid)

    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = Pool()
        with patch(BACKEND, return_value=backend(pool)):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])
        self.assertIsNone(view.hybrid)
        self.assertFalse(hasattr(view, "_backend"))

        # API v1 reads optional operations lazily: a consumer that only needs the
        # metadata and pool must not require every backend method to exist.
        minimal = SimpleNamespace(
            token_to_kv_pool=pool,
            forward_metadata=SimpleNamespace(core_attn_metadata="minimal"),
        )
        with patch(BACKEND, return_value=minimal):
            minimal_view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(minimal_view.metadata.core_attn_metadata, "minimal")
        self.assertIs(minimal_view.kv_pool.pool, pool)
        with self.assertRaises(AttributeError):
            _ = minimal_view.attention

    def test_a_replaced_source_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        with patch(BACKEND, return_value=attn_backend):
            binding.forward_view()
            self.assertIs(binding.engram_history(), model.engram_hasher.history)
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch(BACKEND, return_value=backend(Pool())):
            binding.forward_view()
        with patch(BACKEND, return_value=backend(Pool())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()

    def test_hybrid_preparation_rejects_replayssm_on_any_kda_layer(self):
        for layer_id in (0, 2):
            with self.subTest(layer_id=layer_id):
                fixture = hybrid_backend()
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                layer_cache = fixture.request_pool.mamba2_layer_cache
                replay_state = layer_cache(layer_id)
                replay_state.replayssm_d = torch.zeros(1)
                with (
                    patch.object(
                        fixture.request_pool,
                        "mamba2_layer_cache",
                        side_effect=lambda current_layer: (
                            replay_state
                            if current_layer == layer_id
                            else layer_cache(current_layer)
                        ),
                    ),
                    self.assertRaisesRegex(
                        ValueError, f"KDA layer {layer_id}.*ReplaySSM"
                    ),
                ):
                    binding.preparation_view(fixture.runner)

    def test_hybrid_preparation_is_typed_stable_and_capacity_is_live(self):
        fixture = hybrid_backend()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        prepared = binding.preparation_view(fixture.runner)
        hybrid = prepared.hybrid
        self.assertIsNotNone(hybrid)
        self.assertEqual(fixture.token_pool.key_buffer_calls, [1, 3])
        self.assertIs(binding.preparation_view(fixture.runner), prepared)
        self.assertEqual(fixture.token_pool.key_buffer_calls, [1, 3, 1, 3])

        self.assertEqual(hybrid.mla.reserved_capacity, 8)
        self.assertEqual(hybrid.mla.serving_capacity, 8)
        self.assertTrue(hybrid.mla.post_capture_active)
        self.assertEqual(hybrid.mla.page_size, 2)
        self.assertIs(hybrid.mla.logical_dtype, torch.float8_e4m3fn)
        self.assertIs(hybrid.mla.storage_dtype, torch.uint8)
        self.assertEqual(dict(hybrid.mla.layer_mapping), {1: 0, 3: 1})
        self.assertEqual(tuple(hybrid.mla.cache(1).shape), (5, 2, 3))
        self.assertIs(hybrid.mla.cache(1).dtype, torch.uint8)
        self.assertEqual(
            hybrid.mla.cache(1).data_ptr(), fixture.full_pool.kv_buffer[0].data_ptr()
        )
        self.assertEqual(
            hybrid.mla.cache(1).storage_offset(),
            fixture.full_pool.kv_buffer[0].storage_offset(),
        )
        self.assertEqual(hybrid.kda.capacity, 5)
        self.assertEqual(dict(hybrid.kda.layer_mapping), {0: 0, 2: 1})
        self.assertEqual(
            hybrid.kda.state(2).temporal.data_ptr(),
            fixture.mamba_pool.temporal[1].data_ptr(),
        )
        with self.assertRaises(TypeError):
            hybrid.mla.layer_mapping[1] = 9
        self.assertFalse(hasattr(hybrid.mla, "pool"))
        self.assertFalse(hasattr(hybrid.kda, "pool"))

        with patch(BACKEND, return_value=fixture.backend):
            no_batch_view = binding.forward_view()
        self.assertIsNone(no_batch_view.hybrid)
        self.assertFalse(hasattr(no_batch_view, "kv_pool"))
        self.assertFalse(hasattr(no_batch_view, "metadata"))

        # Post-capture sizing changes the live capacity, not the reserved tensor
        # identities the graph captured.
        fixture.token_pool.size = 6
        fixture.full_pool.size = 6
        self.assertEqual(hybrid.mla.reserved_capacity, 8)
        self.assertEqual(hybrid.mla.serving_capacity, 6)
        with patch(BACKEND, return_value=fixture.backend):
            binding.assert_live()
        self.assertEqual(fixture.token_pool.key_buffer_calls, [1, 3, 1, 3, 1, 3])

    def test_hybrid_preparation_preserves_strided_mla_rows(self):
        fixture = hybrid_backend()
        fixture.full_pool.page_size = 1
        backing = torch.zeros(10, 2, 3, dtype=torch.uint8)
        strided = backing[:, :1, :]
        fixture.full_pool.kv_buffer[0] = strided

        hybrid = (
            mb.MegakernelBinding(model=torch.nn.Module())
            .preparation_view(fixture.runner)
            .hybrid
        )
        cache = hybrid.mla.cache(1)
        self.assertEqual(tuple(cache.shape), (10, 1, 3))
        self.assertIs(cache.dtype, torch.uint8)
        self.assertEqual(cache.data_ptr(), strided.data_ptr())
        self.assertEqual(cache.storage_offset(), strided.storage_offset())
        self.assertEqual(cache.stride(1), strided.stride(0))

    def test_hybrid_step_normalizes_metadata_and_delegates_state_tracking(self):
        fixture = hybrid_backend()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        prepared = binding.preparation_view(fixture.runner)
        batch = forward_batch()
        with patch(BACKEND, return_value=fixture.backend):
            view = binding.forward_view(batch)

        self.assertIs(
            view.hybrid.mla.block_table,
            fixture.full.forward_decode_metadata.block_kv_indices,
        )
        self.assertIs(
            view.hybrid.mla.sequence_lengths,
            fixture.full.forward_decode_metadata.seq_lens_k,
        )
        self.assertIs(view.hybrid.mla.write_slots, batch.out_cache_loc)
        self.assertIs(
            view.hybrid.kda.state_indices,
            fixture.linear.forward_metadata.mamba_cache_indices,
        )
        self.assertFalse(hasattr(view, "kv_pool"))
        self.assertFalse(hasattr(view, "metadata"))

        view.hybrid.kda.track_decode_state(2)
        self.assertEqual(len(fixture.tracked), 1)
        args = fixture.tracked[0]
        state = prepared.hybrid.kda.state(2)
        self.assertIs(args[0], batch)
        self.assertIs(args[1], state.conv[0])
        self.assertIs(args[2], state.temporal)
        self.assertIs(args[3], view.hybrid.kda.state_indices)
        self.assertEqual(args[4], 2)

        batch.out_cache_loc_is_physical = False
        with (
            patch(BACKEND, return_value=fixture.backend),
            self.assertRaisesRegex(RuntimeError, "write slots are not physical"),
        ):
            binding.forward_view(batch)

    def test_capture_metadata_allows_content_updates_but_not_replacement(self):
        fixture = hybrid_backend()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        binding.preparation_view(fixture.runner)
        batch = forward_batch()
        with patch(BACKEND, return_value=fixture.backend):
            first = binding.forward_view(batch)
            binding.seal_capture_view(batch, first)
            first.hybrid.mla.block_table.add_(1)
            first.hybrid.mla.sequence_lengths.add_(1)
            first.hybrid.mla.write_slots.add_(1)
            first.hybrid.kda.state_indices.add_(1)
            binding.seal_capture_view(batch, binding.forward_view(batch))
            binding.assert_live()

        replacements = {
            "mla.block_table": lambda f, b: setattr(
                f.full.forward_decode_metadata,
                "block_kv_indices",
                torch.zeros_like(f.full.forward_decode_metadata.block_kv_indices),
            ),
            "mla.sequence_lengths": lambda f, b: setattr(
                f.full.forward_decode_metadata,
                "seq_lens_k",
                torch.zeros_like(f.full.forward_decode_metadata.seq_lens_k),
            ),
            "mla.write_slots": lambda f, b: setattr(
                b, "out_cache_loc", torch.zeros_like(b.out_cache_loc)
            ),
            "kda.state_indices": lambda f, b: setattr(
                f.linear.forward_metadata,
                "mamba_cache_indices",
                torch.zeros_like(f.linear.forward_metadata.mamba_cache_indices),
            ),
        }
        for expected, replace in replacements.items():
            for before_seal in (False, True):
                with self.subTest(expected=expected, before_seal=before_seal):
                    fixture = hybrid_backend()
                    binding = mb.MegakernelBinding(model=torch.nn.Module())
                    binding.preparation_view(fixture.runner)
                    batch = forward_batch()
                    with patch(BACKEND, return_value=fixture.backend):
                        view = binding.forward_view(batch)
                        if before_seal:
                            replace(fixture, batch)
                            with self.assertRaisesRegex(
                                RuntimeError, f"{expected} changed before seal"
                            ):
                                binding.seal_capture_view(batch, view)
                        else:
                            binding.seal_capture_view(batch, view)
                            replace(fixture, batch)
                            with self.assertRaisesRegex(RuntimeError, expected):
                                binding.assert_live()

    def test_required_metadata_fields_are_distinct_from_unset_values(self):
        fields = (
            ("mla", "block_kv_indices"),
            ("mla", "seq_lens_k"),
            ("batch", "out_cache_loc"),
            ("kda", "mamba_cache_indices"),
        )
        for owner, field in fields:
            for missing in (False, True):
                with self.subTest(owner=owner, field=field, missing=missing):
                    fixture = hybrid_backend()
                    binding = mb.MegakernelBinding(model=torch.nn.Module())
                    binding.preparation_view(fixture.runner)
                    batch = forward_batch()
                    target = {
                        "mla": fixture.full.forward_decode_metadata,
                        "kda": fixture.linear.forward_metadata,
                        "batch": batch,
                    }[owner]
                    if missing:
                        delattr(target, field)
                        error, message = AttributeError, field
                    else:
                        setattr(target, field, None)
                        error, message = RuntimeError, "metadata is incomplete"
                    with (
                        patch(BACKEND, return_value=fixture.backend),
                        self.assertRaisesRegex(error, message),
                    ):
                        binding.forward_view(batch)

    def test_uninitialized_metadata_is_rejected(self):
        for kind in ("MLA", "KDA"):
            with self.subTest(kind=kind):
                fixture = hybrid_backend()
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                binding.preparation_view(fixture.runner)
                if kind == "MLA":
                    fixture.full.forward_decode_metadata = None
                else:
                    fixture.linear.forward_metadata = None
                with (
                    patch(BACKEND, return_value=fixture.backend),
                    self.assertRaisesRegex(
                        RuntimeError, f"{kind} decode metadata is unavailable"
                    ),
                ):
                    binding.forward_view(forward_batch())

    def test_capture_keeps_each_metadata_owner(self):
        fixture = hybrid_backend()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        binding.preparation_view(fixture.runner)
        first_batch = forward_batch()
        first_kda = fixture.linear.forward_metadata
        with patch(BACKEND, return_value=fixture.backend):
            binding.seal_capture_view(first_batch, binding.forward_view(first_batch))
            fixture.full.forward_decode_metadata = SimpleNamespace(
                block_kv_indices=torch.zeros(1, 4, dtype=torch.int32),
                seq_lens_k=torch.zeros(1, dtype=torch.int32),
            )
            fixture.linear.forward_metadata = SimpleNamespace(
                mamba_cache_indices=torch.zeros(1, dtype=torch.int32)
            )
            second_batch = forward_batch()
            second_batch.batch_size = 1
            second_batch.out_cache_loc = torch.zeros(1, dtype=torch.int32)
            binding.seal_capture_view(second_batch, binding.forward_view(second_batch))
            binding.assert_live()

            first_kda.mamba_cache_indices = torch.zeros_like(
                first_kda.mamba_cache_indices
            )
            with self.assertRaisesRegex(RuntimeError, "kda.state_indices"):
                binding.assert_live()

    def test_each_hybrid_source_family_is_sealed(self):
        cases = [
            (
                "hybrid_backend",
                lambda f: setattr(f.runner, "attn_backend", object()),
            ),
            (
                "full_backend",
                lambda f: setattr(f.backend, "full_attn_backend", SimpleNamespace()),
            ),
            (
                "linear_backend",
                lambda f: setattr(f.backend, "linear_attn_backend", SimpleNamespace()),
            ),
            (
                "hybrid_kv_pool",
                lambda f: setattr(f.backend, "token_to_kv_pool", object()),
            ),
            (
                "full_kv_pool",
                lambda f: setattr(f.token_pool, "full_kv_pool", FullPool()),
            ),
            (
                "request_pool",
                lambda f: setattr(f.backend, "req_to_token_pool", object()),
            ),
            (
                "mamba_pool",
                lambda f: setattr(f.request_pool, "mamba_pool", MambaPool()),
            ),
            (
                "attention_backend_list",
                lambda f: f.backend.attn_backend_list.__setitem__(0, SimpleNamespace()),
            ),
            (
                "full_backend_kv_pool",
                lambda f: setattr(f.full, "token_to_kv_pool", TokenPool()),
            ),
            (
                "full_backend_request_pool",
                lambda f: setattr(f.full, "req_to_token_pool", RequestPool()),
            ),
            (
                "linear_backend_kv_pool",
                lambda f: setattr(f.linear, "token_to_kv_pool", TokenPool()),
            ),
            (
                "linear_backend_request_pool",
                lambda f: setattr(f.linear, "req_to_token_pool", RequestPool()),
            ),
            (
                "full_layer_mapping",
                lambda f: setattr(
                    f.token_pool,
                    "full_attention_layer_id_mapping",
                    dict(f.token_pool.full_attention_layer_id_mapping),
                ),
            ),
            (
                "full_layer_mapping",
                lambda f: f.token_pool.full_attention_layer_id_mapping.__setitem__(
                    5, 2
                ),
            ),
            (
                "mamba_layer_mapping",
                lambda f: setattr(
                    f.request_pool, "mamba_map", dict(f.request_pool.mamba_map)
                ),
            ),
            (
                "mamba_layer_mapping",
                lambda f: f.request_pool.mamba_map.__setitem__(4, 2),
            ),
            (
                "mla_cache",
                lambda f: f.full_pool.kv_buffer.__setitem__(
                    0, torch.zeros_like(f.full_pool.kv_buffer[0])
                ),
            ),
            (
                "kda_conv",
                lambda f: f.mamba_pool.conv.__setitem__(
                    0, torch.zeros_like(f.mamba_pool.conv[0])
                ),
            ),
            (
                "kda_temporal",
                lambda f: setattr(
                    f.mamba_pool,
                    "temporal",
                    torch.zeros_like(f.mamba_pool.temporal),
                ),
            ),
        ]
        for expected, replace in cases:
            with self.subTest(expected=expected):
                fixture = hybrid_backend()
                binding = mb.MegakernelBinding(model=torch.nn.Module())
                binding.preparation_view(fixture.runner)
                replace(fixture)
                with self.assertRaisesRegex(RuntimeError, expected):
                    binding.assert_live()


if __name__ == "__main__":
    unittest.main()
