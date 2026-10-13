# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multimodal weight-loading utilities."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    checkpoint_bytes,
    dit_parameter_count,
    load_safetensors_state_dict,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    _disable_runai_streamer_rank_discovery_collective,
    get_lock,
)
from sglang.multimodal_gen.runtime.weights.source import (
    filter_duplicate_precision_variant_safetensors,
)
from sglang.test.test_utils import CustomTestCase

_DIST_STREAMER_MOD = "runai_model_streamer.distributed_streamer.distributed_streamer"


class TestPrecisionVariantSelection(unittest.TestCase):
    def test_prefers_canonical_family_across_shard_layouts(self):
        files = [
            "/tmp/model.safetensors",
            "/tmp/model.fp16-00001-of-00002.safetensors",
            "/tmp/model.fp16-00002-of-00002.safetensors",
            "/tmp/other.bf16.safetensors",
        ]

        self.assertEqual(
            filter_duplicate_precision_variant_safetensors(files),
            ["/tmp/model.safetensors", "/tmp/other.bf16.safetensors"],
        )

    def test_shared_state_dict_loader_uses_canonical_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            canonical = model_dir / "model.safetensors"
            variant = model_dir / "model.fp16.safetensors"
            safetensors_save_file({"weight": torch.tensor([1.0])}, canonical)
            safetensors_save_file({"weight": torch.tensor([2.0])}, variant)

            state_dict = load_safetensors_state_dict(str(model_dir))

        self.assertTrue(torch.equal(state_dict["weight"], torch.tensor([1.0])))

    def test_index_selection_precedes_canonical_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            canonical = model_dir / "model.safetensors"
            variant = model_dir / "model.fp16.safetensors"
            index = model_dir / "model.safetensors.index.json"
            canonical.touch()
            variant.touch()
            index.write_text('{"weight_map":{"weight":"model.fp16.safetensors"}}')

            selected = _list_safetensors_files(str(model_dir), index_file=index.name)

        self.assertEqual(selected, [str(variant)])

    def test_raw_candidates_preserve_explicit_precision_choice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            canonical = model_dir / "model.safetensors"
            variant = model_dir / "model.fp16.safetensors"
            canonical.touch()
            variant.touch()

            selected = _list_safetensors_files(str(model_dir), raw_candidates=True)

        self.assertEqual(selected, [str(variant), str(canonical)])

    def test_precision_only_index_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            shard = model_dir / "model.fp16-00001-of-00001.safetensors"
            index = model_dir / "model.fp16.safetensors.index.json"
            shard.touch()
            index.write_text(
                '{"weight_map":{"weight":"model.fp16-00001-of-00001.safetensors"}}'
            )

            selected = _list_safetensors_files(str(model_dir))

        self.assertEqual(selected, [str(shard)])

    def test_checkpoint_bytes_counts_only_selected_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            canonical = model_dir / "model.safetensors"
            variant = model_dir / "model.fp16.safetensors"
            canonical.write_bytes(b"a" * 17)
            variant.write_bytes(b"b" * 11)

            self.assertEqual(checkpoint_bytes(str(model_dir)), 17)

    def test_checkpoint_bytes_supports_explicit_file(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint.write(b"checkpoint")
            checkpoint.flush()

            self.assertEqual(checkpoint_bytes(checkpoint.name), 10)


_LOADER_UTILS = "sglang.multimodal_gen.runtime.loader.utils"
_SNAPSHOT_DOWNLOAD = f"{_LOADER_UTILS}.snapshot_download"
_TWO_SHARD_INDEX = (
    '{"weight_map":{"a":"shard-1.safetensors","b":"shard-2.safetensors"}}'
)


def _write_sharded_transformer(root: Path) -> None:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    safetensors_save_file({"a": torch.zeros(3, 4)}, transformer / "shard-1.safetensors")
    safetensors_save_file(
        {"b": torch.zeros(5, dtype=torch.bfloat16)},
        transformer / "shard-2.safetensors",
    )
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        _TWO_SHARD_INDEX
    )


class TestDitParameterCount(CustomTestCase):
    def test_counts_every_indexed_shard_across_float_dtypes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sharded_transformer(Path(tmpdir) / "sub")

            self.assertEqual(
                dit_parameter_count(tmpdir, subfolder="sub", revision=None), 17
            )

    def test_counts_the_canonical_file_over_a_precision_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = Path(tmpdir) / "transformer"
            transformer.mkdir()
            safetensors_save_file(
                {"w": torch.zeros(2, 3)},
                transformer / "diffusion_pytorch_model.safetensors",
            )
            safetensors_save_file(
                {"w": torch.zeros(2, 3, dtype=torch.float16)},
                transformer / "diffusion_pytorch_model.fp16.safetensors",
            )

            self.assertEqual(
                dit_parameter_count(tmpdir, subfolder=None, revision=None), 6
            )

    def test_expands_a_home_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sharded_transformer(Path(tmpdir) / "model")
            with patch.dict(os.environ, {"HOME": tmpdir}):
                self.assertEqual(
                    dit_parameter_count("~/model", subfolder=None, revision=None), 17
                )

    def test_missing_shard_or_quantized_weights_leave_the_dit_unsized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_shard = Path(tmpdir) / "missing" / "transformer"
            missing_shard.mkdir(parents=True)
            (
                missing_shard / "diffusion_pytorch_model.safetensors.index.json"
            ).write_text(_TWO_SHARD_INDEX)
            quantized = Path(tmpdir) / "quantized" / "transformer"
            quantized.mkdir(parents=True)
            safetensors_save_file(
                {
                    "w": torch.zeros(2, 3, dtype=torch.float8_e4m3fn),
                    "scale": torch.ones(1),
                },
                quantized / "diffusion_pytorch_model.safetensors",
            )

            for model_dir in ("missing", "quantized"):
                with self.subTest(model_dir=model_dir):
                    self.assertIsNone(
                        dit_parameter_count(
                            os.path.join(tmpdir, model_dir),
                            subfolder=None,
                            revision=None,
                        )
                    )

    def test_single_file_checkpoint_is_not_sized(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            snapshot_download = Mock()
            with patch(_SNAPSHOT_DOWNLOAD, snapshot_download):
                self.assertIsNone(
                    dit_parameter_count(checkpoint.name, subfolder=None, revision=None)
                )

        snapshot_download.assert_not_called()

    def test_cached_hub_snapshot_is_sized_without_the_network(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_sharded_transformer(Path(tmpdir))
            hub_download = Mock()
            with (
                patch(_SNAPSHOT_DOWNLOAD, return_value=tmpdir) as snapshot_download,
                patch(f"{_LOADER_UTILS}.hf_hub_download", hub_download),
            ):
                self.assertEqual(
                    dit_parameter_count("org/model", subfolder=None, revision="abc"),
                    17,
                )

        hub_download.assert_not_called()
        self.assertTrue(snapshot_download.call_args.kwargs["local_files_only"])
        self.assertEqual(snapshot_download.call_args.kwargs["revision"], "abc")

    def test_hub_headers_size_an_uncached_or_partly_cached_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = Path(tmpdir) / "index.json"
            index.write_text(_TWO_SHARD_INDEX)
            # a cached snapshot whose second shard never finished downloading
            partial = Path(tmpdir) / "partial"
            _write_sharded_transformer(partial)
            (partial / "transformer" / "shard-2.safetensors").unlink()
            header = SimpleNamespace(parameter_count={"F32": 10, "BF16": 2})
            for snapshot in (
                Mock(side_effect=LocalEntryNotFoundError("not cached")),
                Mock(return_value=str(partial)),
            ):
                with (
                    self.subTest(snapshot=snapshot),
                    patch.dict(os.environ, {"SGLANG_USE_MODELSCOPE": "0"}),
                    patch(_SNAPSHOT_DOWNLOAD, snapshot),
                    patch(f"{_LOADER_UTILS}.hf_hub_download", return_value=str(index)),
                    patch(
                        f"{_LOADER_UTILS}.parse_safetensors_file_metadata",
                        return_value=header,
                    ) as parse,
                ):
                    self.assertEqual(
                        dit_parameter_count(
                            "org/model", subfolder=None, revision="abc"
                        ),
                        24,
                    )
                    self.assertEqual(
                        [call.kwargs["filename"] for call in parse.call_args_list],
                        [
                            "transformer/shard-1.safetensors",
                            "transformer/shard-2.safetensors",
                        ],
                    )
                    self.assertEqual(parse.call_args.kwargs["revision"], "abc")

    def test_unreachable_hub_or_modelscope_leaves_the_dit_unsized(self):
        for use_modelscope, hub_download in (
            ("0", Mock(side_effect=LocalEntryNotFoundError("unreachable"))),
            ("1", Mock()),
        ):
            with (
                self.subTest(use_modelscope=use_modelscope),
                patch.dict(os.environ, {"SGLANG_USE_MODELSCOPE": use_modelscope}),
                patch(_SNAPSHOT_DOWNLOAD, side_effect=LocalEntryNotFoundError("miss")),
                patch(f"{_LOADER_UTILS}.hf_hub_download", hub_download),
                patch(f"{_LOADER_UTILS}.parse_safetensors_file_metadata") as parse,
            ):
                self.assertIsNone(
                    dit_parameter_count("org/model", subfolder=None, revision=None)
                )
                parse.assert_not_called()
                self.assertEqual(hub_download.called, use_modelscope == "0")


class TestDisableRunaiStreamerRankDiscoveryCollective(unittest.TestCase):
    def test_never_touches_torch_distributed_even_when_initialized(self):
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        _disable_runai_streamer_rank_discovery_collective()

        with (
            patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=True),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_world_size", return_value=2),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_rank", return_value=1),
            patch(f"{_DIST_STREAMER_MOD}.dist.new_group") as mock_new_group,
            patch(f"{_DIST_STREAMER_MOD}.dist.all_gather_object") as mock_all_gather,
            patch(f"{_DIST_STREAMER_MOD}.dist.destroy_process_group") as mock_destroy,
        ):
            result = _distributedStreamerParams().find_local_ranks()

        mock_new_group.assert_not_called()
        mock_all_gather.assert_not_called()
        mock_destroy.assert_not_called()
        # rank is still reported correctly -- only the collective is gone
        self.assertEqual(result, (1, 1, [[1]]))

    def test_reports_rank_zero_when_not_distributed(self):
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        _disable_runai_streamer_rank_discovery_collective()

        with patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=False):
            result = _distributedStreamerParams().find_local_ranks()

        self.assertEqual(result, (1, 0, [[0]]))

    def test_idempotent_across_repeated_calls(self):
        # Import-time application plus any re-import/re-entry must not stack
        # wrappers or otherwise change behavior.
        _disable_runai_streamer_rank_discovery_collective()
        _disable_runai_streamer_rank_discovery_collective()

        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        with (
            patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=True),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_world_size", return_value=4),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_rank", return_value=3),
            patch(f"{_DIST_STREAMER_MOD}.dist.new_group") as mock_new_group,
        ):
            result = _distributedStreamerParams().find_local_ranks()

        mock_new_group.assert_not_called()
        self.assertEqual(result, (1, 3, [[3]]))

    def test_noop_when_library_missing_attribute(self):
        # Defensive path: if a future runai_model_streamer release renames or
        # removes find_local_ranks, patching must skip (with a warning) rather
        # than crash import.
        import sglang.multimodal_gen.runtime.loader.weight_utils as wu

        class _StubParams:
            pass

        with patch(f"{_DIST_STREAMER_MOD}._distributedStreamerParams", _StubParams):
            wu._disable_runai_streamer_rank_discovery_collective()  # must not raise

        self.assertFalse(hasattr(_StubParams, "find_local_ranks"))


class TestDiffusionWeightLock(unittest.TestCase):
    def test_long_snapshot_path_uses_bounded_lock_filename(self):
        component_path = os.path.join(
            "/scratch",
            "models--" + "very-long-repository-name-" * 8,
            "snapshots",
            "a" * 64,
            "transformer",
            "config.json",
        )

        with tempfile.TemporaryDirectory() as lock_dir:
            lock = get_lock(component_path, lock_dir)
            lock_filename = os.path.basename(lock.lock_file)

            self.assertLessEqual(len(os.fsencode(lock_filename)), 255)
            with lock:
                self.assertTrue(os.path.exists(lock.lock_file))


if __name__ == "__main__":
    unittest.main()
