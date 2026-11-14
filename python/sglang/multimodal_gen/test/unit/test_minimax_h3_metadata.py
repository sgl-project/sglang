# SPDX-License-Identifier: Apache-2.0
"""Invalid H3 sigma metadata must fail before fetching component weights."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError

from sglang.multimodal_gen.configs.minimax_h3_metadata import (
    validate_minimax_h3_sigma_shift_scales,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    FastH3Pipeline,
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_vdn_pipeline import (
    VDNH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    BUILTIN_MODEL_OVERLAY_REGISTRY,
    MODEL_OVERLAY_METADATA_PATTERNS,
)
from sglang.test.test_utils import CustomTestCase

BASE = "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base"
H3 = "sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline"


def model_index(partition="fl2va"):
    return {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.35.0",
        "transformer": ["sglang", "MiniMaxH3"],
        "_minimax_h3": {
            "schema_version": 1,
            "partition": partition,
            "tasks": ["t2va"] if partition == "fl2va" else ["ref2va"],
            "sigma_shift_scales": {"video": 12, "audio": 3.5},
        },
    }


def pipeline(path, subfolder=None, variant=None, cls=MiniMaxH3Pipeline):
    obj = cls.__new__(cls)
    obj.model_path = str(path)
    obj.server_args = SimpleNamespace(
        model_subfolder=subfolder,
        model_variant=variant,
        revision="test-revision",
        component_weights_paths={"transformer": "merged.safetensors"},
        transformer_weights_path=None,
    )
    return obj


def write_index(root, index):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(index))
    (root / "transformer").mkdir(exist_ok=True)


class TestMiniMaxH3Metadata(CustomTestCase):
    def test_inherited_pipelines_validate_overlay_before_weight_loading(self):
        # Source repos have no model_index; the registered overlay supplies it.
        for cls, repo in (
            (
                FastH3Pipeline,
                "FastVideo/FastVideo-FastH3-8-Step-V2",
            ),
            (VDNH3Pipeline, "OpenVDN/vdn-minimax-h3"),
        ):
            for remote in (False, True):
                for scale in (12.0, float("nan"), True):
                    with (
                        self.subTest(pipeline=cls.__name__, remote=remote, scale=scale),
                        tempfile.TemporaryDirectory() as temp,
                    ):
                        root = Path(temp)
                        source = root / repo.rsplit("/", 1)[-1]
                        source.mkdir()
                        overlay = root / "overlay"
                        index = model_index()
                        index["_minimax_h3"]["sigma_shift_scales"]["video"] = scale
                        write_index(overlay, index)
                        spec = BUILTIN_MODEL_OVERLAY_REGISTRY[repo]

                        def metadata_download(**kwargs):
                            self.assertEqual(kwargs["repo_id"], spec["overlay_repo_id"])
                            self.assertEqual(
                                kwargs["revision"], spec["overlay_revision"]
                            )
                            self.assertEqual(
                                kwargs["allow_patterns"],
                                MODEL_OVERLAY_METADATA_PATTERNS,
                            )
                            return str(overlay)

                        obj = pipeline(repo if remote else source, cls=cls)
                        with (
                            patch(
                                f"{H3}.snapshot_download", side_effect=metadata_download
                            ),
                            patch(
                                f"{H3}.hf_hub_download",
                                side_effect=AssertionError(
                                    "source index must not be fetched"
                                ),
                            ),
                            patch(
                                f"{BASE}.maybe_download_model",
                                return_value=str(overlay),
                            ) as download,
                        ):
                            if scale == 12.0:
                                self.assertEqual(obj._load_config(), index)
                                self.assertEqual(
                                    obj.release_metadata.video_sigma_shift, 12.0
                                )
                            else:
                                with self.assertRaisesRegex(
                                    ValueError, "sigma_shift_scales.video"
                                ):
                                    obj._load_config()
                                download.assert_not_called()

    def test_parser_rejects_invalid_scales_before_float_coercion(self):
        for modality in ("video", "audio"):
            for value in (
                float("nan"),
                float("inf"),
                -float("inf"),
                0,
                -1,
                -0.0,
                True,
                False,
                "12.0",
                None,
                [],
                {},
                10**400,
            ):
                with self.subTest(modality=modality, value=value):
                    index = model_index()
                    index["_minimax_h3"]["sigma_shift_scales"][modality] = value
                    for parse in (
                        validate_minimax_h3_sigma_shift_scales,
                        MiniMaxH3ReleaseMetadata.from_model_index,
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            re.escape(
                                f"model_index.json._minimax_h3.sigma_shift_scales.{modality}"
                            ),
                        ):
                            parse(index)
            index = model_index()
            del index["_minimax_h3"]["sigma_shift_scales"][modality]
            with self.assertRaisesRegex(ValueError, modality):
                MiniMaxH3ReleaseMetadata.from_model_index(index)

    def test_preflight_rejects_invalid_metadata_without_downloading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in (
                float("nan"),
                float("inf"),
                -float("inf"),
                0,
                -1,
                True,
                False,
                "12.0",
                10**400,
            ):
                for modality in ("video", "audio"):
                    with self.subTest(value=value, modality=modality):
                        index = model_index()
                        index["_minimax_h3"]["sigma_shift_scales"][modality] = value
                        write_index(root / "FL2VA", index)
                        for remote in (False, True):
                            obj = pipeline("owner/h3" if remote else root)
                            with (
                                patch(
                                    f"{H3}.hf_hub_download",
                                    return_value=str(root / "FL2VA/model_index.json"),
                                ),
                                patch(
                                    f"{BASE}.maybe_download_model",
                                    side_effect=AssertionError(
                                        "weight download reached"
                                    ),
                                ) as download,
                                self.assertRaisesRegex(ValueError, modality),
                            ):
                                obj._load_config()
                            download.assert_not_called()

    def test_selected_index_matches_weight_directory_and_revision(self):
        for remote, subfolder, variant, selected in (
            (False, None, None, ""),
            (False, None, None, "FL2VA"),
            (False, "custom", None, "custom"),
            (True, None, None, "FL2VA"),
            (True, ".", None, "."),
            (True, "custom", None, "custom"),
            (True, None, "ref2va", "Ref2VA"),
            (True, None, "hybrid", "Ref2VA"),
        ):
            with (
                self.subTest(
                    remote=remote,
                    subfolder=subfolder,
                    variant=variant,
                    selected=selected,
                ),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                index = model_index("ref2va" if selected == "Ref2VA" else "fl2va")
                write_index(root / selected, index)
                obj = pipeline("owner/h3" if remote else root, subfolder, variant)
                with (
                    patch(
                        f"{H3}.hf_hub_download",
                        return_value=str(root / selected / "model_index.json"),
                    ) as fetch,
                    patch(
                        f"{BASE}.maybe_download_model", return_value=str(root)
                    ) as download,
                ):
                    self.assertEqual(obj._load_config(), index)
                self.assertEqual(Path(obj.model_path), root / selected)
                self.assertEqual(
                    obj.release_metadata.sigma_shift_scales,
                    {"video": 12.0, "audio": 3.5},
                )
                self.assertIsInstance(obj.release_metadata.video_sigma_shift, float)
                self.assertEqual(download.call_args.kwargs["revision"], "test-revision")
                if selected:
                    self.assertEqual(
                        download.call_args.kwargs["allow_patterns"], [f"{selected}/**"]
                    )
                if remote:
                    fetch.assert_called_once_with(
                        repo_id="owner/h3",
                        filename=f"{selected}/model_index.json",
                        revision="test-revision",
                    )
                else:
                    fetch.assert_not_called()

    def test_missing_or_damaged_index_does_not_fall_back_to_weight_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for content, error in (
                (None, FileNotFoundError),
                ("{", ValueError),
                ("[]", ValueError),
                ("{}", ValueError),
            ):
                index_path = root / "model_index.json"
                if content is not None:
                    index_path.write_text(content)
                with (
                    self.subTest(content=content),
                    patch(f"{BASE}.maybe_download_model") as download,
                ):
                    with self.assertRaises(error):
                        pipeline(root, ".")._load_config()
                    download.assert_not_called()
            with (
                patch(
                    f"{H3}.hf_hub_download",
                    side_effect=EntryNotFoundError("missing index"),
                ),
                patch(f"{BASE}.maybe_download_model") as download,
                self.assertRaises(EntryNotFoundError),
            ):
                pipeline("owner/h3")._load_config()
            download.assert_not_called()

    def test_invalid_subfolder_fails_before_metadata_fetch(self):
        with (
            patch(f"{H3}.hf_hub_download") as fetch,
            patch(f"{BASE}.maybe_download_model") as download,
        ):
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                pipeline("owner/h3", "../escape")._load_config()
            fetch.assert_not_called()
            download.assert_not_called()

    def test_default_hook_does_not_fetch_metadata_for_other_pipelines(self):
        # Exercise the real base loading path without constructing model modules.
        class OtherPipeline(ComposedPipelineBase):
            def create_pipeline_stages(self, server_args):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = model_index()
            del index["_minimax_h3"]
            write_index(root, index)
            obj = OtherPipeline.__new__(OtherPipeline)
            obj.model_path = str(root)
            obj.server_args = SimpleNamespace(model_subfolder=None, revision=None)
            with (
                patch(
                    f"{H3}.hf_hub_download",
                    side_effect=AssertionError("unexpected metadata fetch"),
                ),
                patch(f"{BASE}.maybe_download_model", return_value=str(root)),
            ):
                self.assertEqual(obj._load_config(), index)


if __name__ == "__main__":
    unittest.main()
