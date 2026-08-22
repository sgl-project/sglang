import importlib.util
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashMLASharedDemandCacheSource(unittest.TestCase):
    @staticmethod
    def _patch_text() -> str:
        repo_root = Path(__file__).resolve().parents[5]
        return (
            repo_root
            / "python/sglang/kernels/aot/cmake/flashmla_shared_demand_cache.patch"
        ).read_text()

    @classmethod
    def _file_patch(cls, path: str) -> str:
        source = cls._patch_text()
        begin = source.index(f"a/{path}")
        next_file = source.find("\ndiff -ruN ", begin)
        return source[begin:] if next_file < 0 else source[begin:next_file]

    def test_row_subgroup_broadcasts_access_and_slot(self):
        adapter = self._file_patch(
            "csrc/sm90/decode/sparse_fp8/components/shared_kv_adapter.h"
        )

        self.assertNotIn("packed_probe_result", adapter)
        self.assertEqual(adapter.count("__shfl_sync"), 2)

    def test_probe_distinguishes_all_remote_read_reasons(self):
        header = self._patch_text()

        self.assertIn("FILLING_FALLBACK", header)
        self.assertIn("COLLISION_FALLBACK", header)

    def test_persistent_decode_can_promote_same_row_across_generations(self):
        source = self._patch_text()

        self.assertIn("bool allow_stale_ready_hit", source)
        self.assertIn("allow_stale_ready_hit && same_row", source)
        self.assertIn(
            "params.shared_kv_cache_generation != nullptr",
            source,
        )

    def test_cached_rows_use_32_byte_aligned_stride(self):
        header = self._patch_text()

        self.assertIn("ROW_BYTES = 656", header)

    def test_shared_demand_cache_header_closes_its_namespace(self):
        patch_text = self._patch_text()
        header_begin = patch_text.index(
            "a/csrc/sm90/decode/sparse_fp8/components/shared_demand_cache.h"
        )
        header_end = patch_text.index(
            "a/csrc/sm90/decode/sparse_fp8/config.h", header_begin
        )
        header_patch = patch_text[header_begin:header_end]

        self.assertIn(
            "+}  // namespace sm90::decode::sparse_fp8::shared_demand_cache",
            header_patch,
        )

    def test_shared_demand_cache_header_defines_ready_publication(self):
        patch_text = self._patch_text()
        header_begin = patch_text.index(
            "a/csrc/sm90/decode/sparse_fp8/components/shared_demand_cache.h"
        )
        header_end = patch_text.index(
            "a/csrc/sm90/decode/sparse_fp8/config.h", header_begin
        )
        header_patch = patch_text[header_begin:header_end]

        self.assertIn("void publish_ready(", header_patch)
        self.assertIn("store_release(tags + slot", header_patch)

    def test_slot_mapping_matches_resident_shift13_hash(self):
        header = self._patch_text()

        self.assertIn("physical_row ^ (physical_row >> 13)", header)
        self.assertIn("2654435761u", header)

    def test_fixed_request_slot_uses_one_capacity_wide_probe(self):
        adapter = self._file_patch(
            "csrc/sm90/decode/sparse_fp8/components/shared_kv_adapter.h"
        )

        self.assertIn("params.shared_kv_cache_rows_per_request", adapter)
        self.assertIn(
            "(request_slot - 1) * params.shared_kv_cache_rows_per_request",
            adapter,
        )
        self.assertEqual(adapter.count("shared_demand_cache::probe"), 1)

    def test_split_kernel_delegates_shared_row_policy_to_adapter(self):
        split_kernel = self._file_patch("csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh")

        self.assertIn("shared_kv_adapter::resolve<", split_kernel)
        self.assertIn("shared_row.store_fragment<", split_kernel)
        self.assertIn("shared_row.publish<", split_kernel)
        self.assertNotIn("shared_demand_cache::probe", split_kernel)

    def test_request_slot_metadata_matches_wrapper_and_extension_schema(self):
        repo_root = Path(__file__).resolve().parents[5]
        wrapper = (
            repo_root / "python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py"
        ).read_text()
        extension = (
            repo_root / "python/sglang/kernels/aot/csrc/flashmla_extension.cc"
        ).read_text()

        for name in (
            "shared_kv_request_slots",
            "shared_kv_cache_rows_per_request",
            "shared_kv_num_request_slots",
            "shared_kv_cache_generation_tensor",
        ):
            self.assertIn(name, wrapper)
            self.assertIn(name, extension)

    def test_current_row_width_is_dynamic_in_flashmla_patch(self):
        source = self._patch_text()

        self.assertIn("shared_kv_current_rows->size(1) >= 1", source)
        self.assertIn("shared_kv_current_rows->size(1) <= 4", source)
        self.assertIn(
            "shared_kv_current_row_ids->size(1) == shared_kv_current_rows->size(1)",
            source,
        )
        self.assertNotIn("shared_kv_current_rows->size(1) == 4", source)

    def test_probe_owns_stale_ready_policy_not_tag_encoding(self):
        header = self._patch_text()
        tag_begin = header.index("uint64_t filling_tag(")
        tag_end = header.index(") {", tag_begin)
        probe_begin = header.index("ProbeResult probe(")
        probe_end = header.index(") {", probe_begin)

        self.assertNotIn("allow_stale_ready_hit", header[tag_begin:tag_end])
        self.assertIn("bool allow_stale_ready_hit", header[probe_begin:probe_end])

    def test_unified_patch_hunk_lengths_match_headers(self):
        lines = self._patch_text().splitlines()
        header = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

        for index, line in enumerate(lines):
            match = header.match(line)
            if match is None:
                continue
            expected_old = int(match.group(1) or 1)
            expected_new = int(match.group(2) or 1)
            actual_old = actual_new = 0
            for hunk_line in lines[index + 1 :]:
                if hunk_line.startswith(("@@ ", "diff ")):
                    break
                if hunk_line.startswith((" ", "-")):
                    actual_old += 1
                if hunk_line.startswith((" ", "+")):
                    actual_new += 1
            self.assertEqual(
                (actual_old, actual_new),
                (expected_old, expected_new),
                f"malformed unified-diff hunk: {line}",
            )

    def test_current_row_marker_precedes_local_and_persistent_cache_paths(self):
        adapter = self._file_patch(
            "csrc/sm90/decode/sparse_fp8/components/shared_kv_adapter.h"
        )
        current = adapter.index("token_index <= -2 && token_index >= -5")
        current_source = adapter.index("params.shared_kv_current_rows", current)
        owner_local = adapter.index("const bool remote_row", current_source)
        persistent_probe = adapter.index("shared_demand_cache::probe", owner_local)

        self.assertLess(current, current_source)
        self.assertLess(current_source, owner_local)
        self.assertLess(owner_local, persistent_probe)
        current_block = adapter[current:owner_local]
        self.assertIn("return -token_index - 2", current_block)
        self.assertNotIn("shared_demand_cache::probe", current_block)
        self.assertNotIn("remote_probe_count", current_block)
        self.assertNotIn("publish_ready", current_block)

    def test_owner_translation_fuses_current_row_marker(self):
        repo_root = Path(__file__).resolve().parents[5]
        source = (
            repo_root / "python/sglang/kernels/ops/attention/dsa/transform_index.py"
        ).read_text()

        physical = source.index("physical_slots =")
        marker = source.index("current_row_markers", physical)
        output = source.index("tl.store(", marker)
        self.assertLess(physical, marker)
        self.assertLess(marker, output)
        self.assertIn("-2 - current_row_slot", source[marker:output])

    def test_backend_forwards_graph_stable_current_rows_to_topk_and_flashmla(self):
        repo_root = Path(__file__).resolve().parents[5]
        backend = (
            repo_root / "python/sglang/srt/layers/attention/dsa_backend.py"
        ).read_text()
        self.assertIn("shared_mla_current_rows", backend)
        self.assertIn(
            "metadata.shared_mla_current_rows[shared_current_rows_layer_idx]",
            backend,
        )
        self.assertIn("current_row_locs=", backend)
        self.assertIn("shared_kv_current_rows=", backend)

    def test_current_row_writer_fuses_shadow_without_flat_staging(self):
        repo_root = Path(__file__).resolve().parents[5]
        cache = (
            repo_root / "python/sglang/srt/mem_cache/dsa_cache_shared.py"
        ).read_text()
        setter = cache[
            cache.index("    def set_mla_kv_buffer_with_current_rows(") : cache.index(
                "    def move_kv_cache(",
                cache.index("    def set_mla_kv_buffer_with_current_rows("),
            )
        ]
        self.assertIn("current_rows.encoded_rows", setter)
        self.assertNotIn("build_mla_current_row_shadow_triton(", setter)

    def test_current_row_count_range_assert_is_not_captured_into_cuda_graph(self):
        repo_root = Path(__file__).resolve().parents[5]
        wrapper = (
            repo_root / "python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py"
        ).read_text()
        validator = wrapper[
            wrapper.index("def _validate_shared_kv_current_rows(") : wrapper.index(
                "def flash_mla_with_kvcache(",
                wrapper.index("def _validate_shared_kv_current_rows("),
            )
        ]

        guard = validator.index(
            "if not torch.cuda.is_available() or not "
            "torch.cuda.is_current_stream_capturing():"
        )
        dynamic_assert = validator.index("torch._assert_async(", guard)
        self.assertLess(guard, dynamic_assert)


class TestFlashMLASharedDemandCacheWrapper(unittest.TestCase):
    @staticmethod
    def _wrapper_module():
        repo_root = Path(__file__).resolve().parents[5]
        path = repo_root / "python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py"
        name = "sgl_kernel._request_scoped_flash_mla_test"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_wrapper_forwards_request_scoped_slot_contract(self):
        flash_mla = self._wrapper_module()

        q = torch.empty((2, 1, 64, 576), dtype=torch.bfloat16)
        k_cache = torch.empty((128, 64, 1, 656), dtype=torch.uint8)
        indices = torch.zeros((2, 1, 2048), dtype=torch.int32)
        row_cache = torch.empty((16, 656), dtype=torch.uint8)
        tags = torch.zeros((2, 8), dtype=torch.int64)
        request_slots = torch.tensor([3, 7], dtype=torch.int64)
        generation = torch.tensor(4, dtype=torch.int32)
        current_rows = torch.empty((2, 4, 656), dtype=torch.uint8)
        current_ids = torch.tensor(
            [[0, -1, -1, -1], [64, -1, -1, -1]], dtype=torch.int32
        )
        current_counts = torch.ones((2,), dtype=torch.int32)
        expected_out = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16)
        expected_lse = torch.empty((2, 64, 1), dtype=torch.float32)
        op = MagicMock(return_value=(expected_out, expected_lse, None, None))

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            patch.object(
                torch.ops.sgl_kernel,
                "sparse_decode_shared_fwd",
                SimpleNamespace(default=op),
                create=True,
            ),
        ):
            out, lse = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=k_cache,
                block_table=torch.empty((2, 0), dtype=torch.int32),
                cache_seqlens=torch.full((2,), 2048, dtype=torch.int32),
                head_dim_v=512,
                tile_scheduler_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.zeros((3,), dtype=torch.int32),
                softmax_scale=576**-0.5,
                is_fp8_kvcache=True,
                indices=indices,
                shared_kv_row_cache=row_cache,
                shared_kv_cache_tags=tags,
                shared_kv_request_slots=request_slots,
                shared_kv_cache_rows_per_request=8,
                shared_kv_num_request_slots=2,
                shared_kv_cache_epoch=1,
                shared_kv_cache_generation_tensor=generation,
                shared_kv_local_row_begin=0,
                shared_kv_local_row_end=1024,
                shared_kv_current_rows=current_rows,
                shared_kv_current_row_ids=current_ids,
                shared_kv_current_row_counts=current_counts,
            )

        self.assertIs(out, expected_out)
        self.assertIs(lse, expected_lse)
        args = op.call_args.args
        self.assertIs(args[12], row_cache)
        self.assertIs(args[13], tags)
        self.assertIs(args[14], request_slots)
        self.assertEqual(args[15:17], (8, 2))
        self.assertIs(args[18], generation)
        self.assertIs(args[21], current_rows)
        self.assertIs(args[22], current_ids)
        self.assertIs(args[23], current_counts)

    def test_wrapper_forwards_fixed_request_slot_contract(self):
        flash_mla = self._wrapper_module()

        q = torch.empty((2, 1, 64, 576), dtype=torch.bfloat16)
        row_cache = torch.empty((16, 656), dtype=torch.uint8)
        tags = torch.zeros((2, 8), dtype=torch.int64)
        request_slots = torch.tensor([1, 2], dtype=torch.int64)
        expected_out = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16)
        expected_lse = torch.empty((2, 64, 1), dtype=torch.float32)
        op = MagicMock(return_value=(expected_out, expected_lse, None, None))

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            patch.object(
                torch.ops.sgl_kernel,
                "sparse_decode_shared_fwd",
                SimpleNamespace(default=op),
                create=True,
            ),
        ):
            out, lse = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=torch.empty((128, 64, 1, 656), dtype=torch.uint8),
                block_table=torch.empty((2, 0), dtype=torch.int32),
                cache_seqlens=torch.full((2,), 2048, dtype=torch.int32),
                head_dim_v=512,
                tile_scheduler_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.zeros((3,), dtype=torch.int32),
                is_fp8_kvcache=True,
                indices=torch.zeros((2, 1, 2048), dtype=torch.int32),
                shared_kv_row_cache=row_cache,
                shared_kv_cache_tags=tags,
                shared_kv_request_slots=request_slots,
                shared_kv_cache_rows_per_request=8,
                shared_kv_num_request_slots=2,
                shared_kv_cache_epoch=1,
                shared_kv_local_row_begin=0,
                shared_kv_local_row_end=1024,
            )

        self.assertIs(out, expected_out)
        self.assertIs(lse, expected_lse)
        args = op.call_args.args
        self.assertEqual(args[15], 8)
        self.assertEqual(args[16], 2)

    def test_wrapper_rejects_partial_or_malformed_current_rows_before_dispatch(self):
        flash_mla = self._wrapper_module()
        q = torch.empty((2, 1, 64, 576), dtype=torch.bfloat16)
        base_kwargs = dict(
            q=q,
            k_cache=torch.empty((128, 64, 1, 656), dtype=torch.uint8),
            block_table=torch.empty((2, 0), dtype=torch.int32),
            cache_seqlens=torch.full((2,), 2048, dtype=torch.int32),
            head_dim_v=512,
            tile_scheduler_metadata=torch.empty((1,), dtype=torch.int32),
            num_splits=torch.zeros((3,), dtype=torch.int32),
            is_fp8_kvcache=True,
            indices=torch.zeros((2, 1, 2048), dtype=torch.int32),
            shared_kv_row_cache=torch.empty((16, 656), dtype=torch.uint8),
            shared_kv_cache_tags=torch.zeros((2, 8), dtype=torch.int64),
            shared_kv_request_slots=torch.tensor([1, 2], dtype=torch.int64),
            shared_kv_cache_rows_per_request=8,
            shared_kv_num_request_slots=2,
            shared_kv_cache_epoch=1,
            shared_kv_local_row_begin=0,
            shared_kv_local_row_end=1024,
        )
        good_rows = torch.empty((2, 4, 656), dtype=torch.uint8)
        good_ids = torch.full((2, 4), -1, dtype=torch.int32)
        good_counts = torch.zeros((2,), dtype=torch.int32)

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            self.assertRaisesRegex(AssertionError, "provided together"),
        ):
            flash_mla.flash_mla_with_kvcache(
                **base_kwargs, shared_kv_current_rows=good_rows
            )

        malformed = (
            (good_rows.float(), good_ids, good_counts, "uint8"),
            (
                torch.empty((2, 4, 1312), dtype=torch.uint8)[:, :, ::2],
                good_ids,
                good_counts,
                "contiguous",
            ),
            (good_rows[:1], good_ids, good_counts, "query rows"),
            (
                torch.empty((2, 0, 656), dtype=torch.uint8),
                torch.empty((2, 0), dtype=torch.int32),
                good_counts,
                "1 <= width <= 4",
            ),
            (
                torch.empty((2, 5, 656), dtype=torch.uint8),
                torch.empty((2, 5), dtype=torch.int32),
                good_counts,
                "1 <= width <= 4",
            ),
            (
                torch.empty((2, 2, 656), dtype=torch.uint8),
                torch.empty((2, 3), dtype=torch.int32),
                good_counts,
                "match",
            ),
            (good_rows, good_ids.long(), good_counts, "int32"),
            (good_rows, good_ids, good_counts.long(), "int32"),
        )
        with patch.object(flash_mla, "_flashmla_import_error", None):
            for rows, ids, counts, message in malformed:
                with self.subTest(message=message), self.assertRaisesRegex(
                    AssertionError, message
                ):
                    flash_mla.flash_mla_with_kvcache(
                        **base_kwargs,
                        shared_kv_current_rows=rows,
                        shared_kv_current_row_ids=ids,
                        shared_kv_current_row_counts=counts,
                    )

            with self.assertRaisesRegex(RuntimeError, "range"):
                flash_mla.flash_mla_with_kvcache(
                    **base_kwargs,
                    shared_kv_current_rows=good_rows,
                    shared_kv_current_row_ids=good_ids,
                    shared_kv_current_row_counts=torch.tensor(
                        [0, 5], dtype=torch.int32
                    ),
                )

    def test_wrapper_accepts_current_row_widths_one_to_four(self):
        flash_mla = self._wrapper_module()
        q = torch.empty((2, 1, 64, 576), dtype=torch.bfloat16)
        expected_out = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16)
        expected_lse = torch.empty((2, 64, 1), dtype=torch.float32)
        op = MagicMock(return_value=(expected_out, expected_lse, None, None))
        base_kwargs = dict(
            q=q,
            k_cache=torch.empty((128, 64, 1, 656), dtype=torch.uint8),
            block_table=torch.empty((2, 0), dtype=torch.int32),
            cache_seqlens=torch.full((2,), 2048, dtype=torch.int32),
            head_dim_v=512,
            tile_scheduler_metadata=torch.empty((1,), dtype=torch.int32),
            num_splits=torch.zeros((3,), dtype=torch.int32),
            is_fp8_kvcache=True,
            indices=torch.zeros((2, 1, 2048), dtype=torch.int32),
            shared_kv_row_cache=torch.empty((16, 656), dtype=torch.uint8),
            shared_kv_cache_tags=torch.zeros((2, 8), dtype=torch.int64),
            shared_kv_request_slots=torch.tensor([1, 2], dtype=torch.int64),
            shared_kv_cache_rows_per_request=8,
            shared_kv_num_request_slots=2,
            shared_kv_cache_epoch=1,
            shared_kv_local_row_begin=0,
            shared_kv_local_row_end=1024,
        )

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            patch.object(
                torch.ops.sgl_kernel,
                "sparse_decode_shared_fwd",
                SimpleNamespace(default=op),
                create=True,
            ),
        ):
            for width in range(1, 5):
                with self.subTest(width=width):
                    out, lse = flash_mla.flash_mla_with_kvcache(
                        **base_kwargs,
                        shared_kv_current_rows=torch.empty(
                            (2, width, 656), dtype=torch.uint8
                        ),
                        shared_kv_current_row_ids=torch.full(
                            (2, width), -1, dtype=torch.int32
                        ),
                        shared_kv_current_row_counts=torch.full(
                            (2,), width, dtype=torch.int32
                        ),
                    )
                    self.assertIs(out, expected_out)
                    self.assertIs(lse, expected_lse)

    def test_wrapper_rejects_multislice_cache_without_request_slots(self):
        flash_mla = self._wrapper_module()

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            self.assertRaisesRegex(AssertionError, "request slots"),
        ):
            flash_mla.flash_mla_with_kvcache(
                q=torch.empty((2, 1, 64, 576), dtype=torch.bfloat16),
                k_cache=torch.empty((128, 64, 1, 656), dtype=torch.uint8),
                block_table=torch.empty((2, 0), dtype=torch.int32),
                cache_seqlens=torch.full((2,), 2048, dtype=torch.int32),
                head_dim_v=512,
                tile_scheduler_metadata=torch.empty((1,), dtype=torch.int32),
                num_splits=torch.zeros((3,), dtype=torch.int32),
                is_fp8_kvcache=True,
                indices=torch.zeros((2, 1, 2048), dtype=torch.int32),
                shared_kv_row_cache=torch.empty((16, 656), dtype=torch.uint8),
                shared_kv_cache_tags=torch.zeros((2, 8), dtype=torch.int64),
                shared_kv_cache_rows_per_request=8,
                shared_kv_num_request_slots=2,
                shared_kv_cache_epoch=1,
                shared_kv_local_row_begin=0,
                shared_kv_local_row_end=1024,
            )


if __name__ == "__main__":
    unittest.main()
