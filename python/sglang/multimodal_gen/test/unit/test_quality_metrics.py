import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sglang.multimodal_gen.benchmarks.bench_image_quality import (
    _check_case_metrics,
    _pair_with_manifest,
    _score_rewards,
    paired_delta,
)
from sglang.multimodal_gen.benchmarks.request_manifest import ManifestRequest
from sglang.multimodal_gen.test import quality_metrics
from sglang.multimodal_gen.test.quality_metrics import (
    QualityScores,
    QualityThresholds,
    check_quality,
    load_quality_thresholds,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase
from sglang.srt.environ import envs

REPO_ROOT = Path(__file__).resolve().parents[5]


def _solid_image(value: int, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _request(index: int, request_id: str) -> ManifestRequest:
    return ManifestRequest(
        request_id=request_id, prompt=f"prompt {index}", sampling_params={}
    )


def test_enforced_metric_without_a_score_is_a_failure():
    """An enforced metric that produced no value must not read as a pass.

    A scorer that returns None on a missing dependency once made the accuracy
    check silently green.
    """
    failures = check_quality(
        scores=QualityScores(clip_score=None),
        thresholds=QualityThresholds(min_clip_score=20.0),
    )
    assert len(failures) == 1
    assert "not computed" in failures[0]


def test_unenforced_metric_is_neither_required_nor_scored():
    # The complement of the case above: a metric with no floor is skipped, so a
    # case must not need every metric installed to enforce one of them.
    assert (
        check_quality(
            scores=QualityScores(clip_score=31.5),
            thresholds=QualityThresholds(min_clip_score=20.0),
        )
        == []
    )
    scores = quality_metrics.compute_quality_scores(
        image=_solid_image(128),
        prompt="a cat",
        thresholds=QualityThresholds(),
    )
    assert scores == QualityScores()


def test_threshold_lookup_falls_back_to_defaults_and_ignores_unknown_cases():
    metadata = {
        "defaults": {"min_clip_score": 15.0},
        "cases": {"case_a": {"min_clip_score": 25.0, "min_clip_iqa": 0.5}},
    }
    case_a = load_quality_thresholds("case_a", metadata)
    assert (case_a.min_clip_score, case_a.min_clip_iqa) == (25.0, 0.5)
    assert case_a.min_image_reward is None

    unknown = load_quality_thresholds("no_such_case", metadata)
    assert unknown.min_clip_score == 15.0
    assert unknown.min_clip_iqa is None


def test_shipped_thresholds_are_loadable_and_enforce_something():
    # A case entry that parses but enforces nothing would turn run_quality_check
    # into a no-op; the harness rejects that at runtime, this catches it earlier.
    metadata = quality_metrics._load_quality_threshold_json()
    assert metadata["cases"], "quality_thresholds.json has no cases"
    for case_id in metadata["cases"]:
        assert load_quality_thresholds(case_id).enforces_anything(), case_id


def test_clip_iqa_pins_data_range_to_the_tensor_scale(monkeypatch):
    """torchmetrics' CLIP-IQA needs data_range to match the input scale.

    It defaults to 1.0 unvalidated, so [0, 255] pixels return noise rather than
    raising; the helper must not leave that to the caller.
    """
    torch = pytest.importorskip("torch")
    recorded = {}

    class FakeMetric:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def __call__(self, tensor):
            recorded["shape"] = tuple(tensor.shape)
            recorded["max_value"] = float(tensor.max())
            return torch.tensor(0.5)

        def reset(self):
            recorded["resets"] = recorded.get("resets", 0) + 1

    fake_multimodal = types.ModuleType("torchmetrics.multimodal")
    fake_multimodal.CLIPImageQualityAssessment = FakeMetric
    fake_root = types.ModuleType("torchmetrics")
    fake_root.multimodal = fake_multimodal
    monkeypatch.setitem(sys.modules, "torchmetrics", fake_root)
    monkeypatch.setitem(sys.modules, "torchmetrics.multimodal", fake_multimodal)
    quality_metrics._clip_iqa_cache.clear()

    try:
        score = quality_metrics.compute_clip_iqa(image=_solid_image(200))
    finally:
        quality_metrics._clip_iqa_cache.clear()

    assert score == pytest.approx(0.5)
    assert recorded["data_range"] == 255.0
    assert recorded["shape"] == (1, 3, 8, 8)
    assert recorded["max_value"] == pytest.approx(200.0)
    # The metric is cached across images, so its accumulated state is dropped
    # per image instead of growing for a whole sweep.
    assert recorded["resets"] == 1


def test_clip_score_reads_the_projected_features_off_the_output_object(monkeypatch):
    """transformers 5 returns an output object from get_image_features.

    Treating it as a tensor -- `.norm(...)`, as torchmetrics 1.9 does -- raises at
    scoring time; the fake exposes only pooler_output, so it raises here instead.
    """
    torch = pytest.importorskip("torch")
    recorded = {}

    class FakeOutput:
        def __init__(self, pooler_output):
            self.pooler_output = pooler_output

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def get_image_features(self, pixel_values):
            recorded["pixel_values"] = pixel_values
            # [3, 4] normalizes to [0.6, 0.8], so the cosine below is 0.6.
            return FakeOutput(torch.tensor([[3.0, 4.0]]))

        def get_text_features(self, input_ids, attention_mask):
            recorded["input_ids"] = input_ids
            recorded["attention_mask"] = attention_mask
            return FakeOutput(torch.tensor([[1.0, 0.0]]))

    class FakeProcessor:
        def __call__(self, *, text, images, **kwargs):
            recorded["text"] = text
            recorded["dtype"] = images[0].dtype
            recorded["shape"] = images[0].shape
            recorded["kwargs"] = kwargs
            return {
                "pixel_values": torch.zeros(1, 3, 8, 8),
                "input_ids": torch.ones(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPModel = types.SimpleNamespace(
        from_pretrained=lambda name: FakeModel()
    )
    fake_transformers.CLIPProcessor = types.SimpleNamespace(
        from_pretrained=lambda name: FakeProcessor()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    quality_metrics._clip_prompt_cache.clear()

    try:
        score = quality_metrics.compute_clip_score(
            image=_solid_image(200), prompt="a cat"
        )
    finally:
        quality_metrics._clip_prompt_cache.clear()

    # 100 * cosine, torchmetrics' scale, and unclamped as torchmetrics leaves it.
    assert score == pytest.approx(60.0)
    assert recorded["text"] == ["a cat"]
    # The processor rescales uint8 [0, 255] HWC itself, while CLIP-IQA wants a
    # float CHW batch; either accepts the other's tensor, so pin the shapes.
    assert recorded["dtype"] == np.uint8
    assert recorded["shape"] == (8, 8, 3)
    assert recorded["kwargs"]["truncation"] is True


def test_image_reward_scores_are_read_from_the_last_json_line():
    # The loader prints a checkpoint banner and tqdm output before the scores.
    stdout = (
        "\n".join(
            [
                "load checkpoint from ImageReward.pt",
                "100%|####|",
                '{"scores": [-0.25, 1.5]}',
            ]
        )
        + "\n"
    )
    assert quality_metrics._parse_image_reward_stdout(stdout) == pytest.approx(
        [-0.25, 1.5]
    )

    with pytest.raises(RuntimeError, match="printed no score"):
        quality_metrics._parse_image_reward_stdout("Traceback ...\n")


def test_a_sweep_launches_the_image_reward_scorer_once(monkeypatch, tmp_path):
    """One launch for the whole directory, not one per image.

    The launch loads a 1.7 GB checkpoint, so a per-image launch spends far more
    time loading than scoring; scores must still land on the right rows.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(json.loads(kwargs["input"]))
        return types.SimpleNamespace(
            returncode=0, stdout='{"scores": [0.5, -0.5, 1.5]}', stderr=""
        )

    monkeypatch.setattr(quality_metrics.subprocess, "run", fake_run)

    rows = [{"index": i, "prompt": f"prompt {i}"} for i in range(3)]
    paths = [tmp_path / f"0000{i}-p.png" for i in range(3)]
    with envs.SGLANG_TEST_IMAGE_REWARD_PYTHON.override("/opt/ir/bin/python"):
        _score_rewards(rows=rows, paths=paths)

    assert len(calls) == 1
    assert calls[0]["pairs"] == [[f"prompt {i}", str(paths[i])] for i in range(3)]
    assert [row["image_reward"] for row in rows] == [0.5, -0.5, 1.5]


def test_image_reward_requires_a_separate_interpreter():
    with envs.SGLANG_TEST_IMAGE_REWARD_PYTHON.override(None):
        with pytest.raises(RuntimeError, match="transformers<5"):
            quality_metrics.compute_image_reward(
                image=_solid_image(128), prompt="a cat"
            )


def test_paired_delta_reports_the_mean_difference_and_its_own_error():
    # Differences of [1, 2, 3, 4]: mean 2.5, sem 0.6454972, z 3.8729833. Pairing is
    # the point -- as independent means the sem would be 0.9128709.
    stats = paired_delta(
        values=[11.0, 12.0, 13.0, 14.0], baseline_values=[10.0, 10.0, 10.0, 10.0]
    )
    assert stats["mean_delta"] == pytest.approx(2.5)
    assert stats["sem"] == pytest.approx(0.6454972, rel=1e-6)
    assert stats["z"] == pytest.approx(3.8729833, rel=1e-6)

    # A constant non-zero difference has no spread to divide by, so z is null
    # rather than 0.0, which would read as "no difference".
    constant = paired_delta(values=[2.0, 2.0], baseline_values=[1.0, 1.0])
    assert (constant["mean_delta"], constant["sem"], constant["z"]) == (1.0, 0.0, None)

    # An all-zero delta really is z=0, not undefined.
    identical = paired_delta(values=[2.0, 2.0], baseline_values=[2.0, 2.0])
    assert (identical["mean_delta"], identical["sem"], identical["z"]) == (
        0.0,
        0.0,
        0.0,
    )


def test_pairing_rejects_images_from_a_different_run(tmp_path):
    requests = [_request(0, "req-a"), _request(1, "req-b")]
    images = [
        (0, "req-a", tmp_path / "00000-req-a.png"),
        (1, "req-c", tmp_path / "00001-req-c.png"),
    ]
    with pytest.raises(ValueError, match="PAIRING ERROR"):
        _pair_with_manifest(images=images, requests=requests)

    paired = _pair_with_manifest(images=images[:1], requests=requests)
    assert [request.request_id for _, request, _ in paired] == ["req-a"]

    # A second output for one row carries an _<output_idx> suffix; that is a
    # generation mode this tool does not pair, not images from another run.
    with pytest.raises(ValueError, match="one image per row"):
        _pair_with_manifest(
            images=[(0, "req-a_1", tmp_path / "00000-req-a_1.png")],
            requests=requests,
        )


def test_case_id_requires_the_metrics_that_case_enforces():
    # check_quality counts an enforced metric with no score as a violation, so a
    # --case-id whose floor was never computed would flag every single image.
    with pytest.raises(ValueError, match="image_reward"):
        _check_case_metrics(
            case_id="flux_2_klein_image_t2i", metrics=("clip_score", "clip_iqa")
        )
    _check_case_metrics(
        case_id="flux_2_klein_image_t2i",
        metrics=("clip_score", "clip_iqa", "image_reward"),
    )


def _registered_diffusion_cases() -> list[DiffusionTestCase]:
    """Cases from the device suites under test/registered/, loaded by path.

    They live outside the sglang package, so only files whose source mentions the
    flag are imported rather than every registered suite.
    """
    cases: list[DiffusionTestCase] = []
    for path in sorted((REPO_ROOT / "test" / "registered").rglob("test_*.py")):
        if "run_quality_check" not in path.read_text(encoding="utf-8"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases.extend(
            case
            for value in vars(module).values()
            if isinstance(value, list)
            for case in value
            if isinstance(case, DiffusionTestCase)
        )
    return cases


def test_quality_checked_cases_have_measured_thresholds():
    # Flipping run_quality_check on without adding an entry makes the case fail
    # mid-run on the first generated image; catch it at collection time instead.
    from sglang.multimodal_gen.test.server.gpu_cases import (
        ONE_GPU_CASES,
        TWO_GPU_CASES,
    )

    # The only case that enables the check today is a registered XPU one, so a
    # gpu_cases-only scan would assert over nothing.
    enabled = [
        case
        for case in [*ONE_GPU_CASES, *TWO_GPU_CASES, *_registered_diffusion_cases()]
        if case.run_quality_check
    ]
    assert enabled, "no case enables run_quality_check; this guard covers nothing"
    for case in enabled:
        assert load_quality_thresholds(case.id).enforces_anything(), (
            f"{case.id} has run_quality_check=True but no floors in "
            f"{quality_metrics.QUALITY_THRESHOLD_PATH}"
        )
