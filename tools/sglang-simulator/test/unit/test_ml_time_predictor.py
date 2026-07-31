import joblib
import pytest
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.ml import MLTimePredictor


class SklearnCompatibleRegressor:
    """Minimal stand-in for GBR, HGB, RF, or another sklearn regressor."""

    def __init__(self, prediction=0.25):
        self.prediction = prediction

    def predict(self, rows):
        assert len(rows) == 1
        assert len(rows[0]) == 18
        return [self.prediction]


def dump_bundle(path, *, features=None, model=None):
    joblib.dump(
        {
            "model": model or SklearnCompatibleRegressor(),
            "features": features or MLTimePredictor.FEATURE_NAMES,
            "target": "iter_latency_seconds",
        },
        path,
    )


def load_predictor(path, latency_scale=1.0):
    return MLTimePredictor(
        model=None,
        hw=None,
        config=None,
        database_path=str(path),
        latency_scale=latency_scale,
    )


def test_ml_predictor_is_model_algorithm_agnostic(tmp_path):
    bundle_path = tmp_path / "regressor.pkl"
    dump_bundle(bundle_path)
    predictor = load_predictor(bundle_path, latency_scale=0.5)
    batch = ScheduleBatch(
        [
            ScheduleRequest(extend_length=8, past_kv_length=16),
            ScheduleRequest(extend_length=1, past_kv_length=32),
        ]
    )

    assert predictor.predict_infer_time(batch) == pytest.approx(0.125)
    assert predictor.predict_infer_time(ScheduleBatch()) == 0.0


def test_ml_predictor_expands_environment_variable(monkeypatch, tmp_path):
    bundle_path = tmp_path / "regressor.pkl"
    dump_bundle(bundle_path)
    monkeypatch.setenv("SGLANG_SIMULATOR_TEST_MODEL_PATH", str(bundle_path))

    predictor = load_predictor("${SGLANG_SIMULATOR_TEST_MODEL_PATH}")

    assert predictor._features == MLTimePredictor.FEATURE_NAMES


def test_ml_predictor_rejects_feature_order_mismatch(tmp_path):
    bundle_path = tmp_path / "wrong-order.pkl"
    features = list(MLTimePredictor.FEATURE_NAMES)
    features[0], features[1] = features[1], features[0]
    dump_bundle(bundle_path, features=features)

    with pytest.raises(ValueError, match="feature contract mismatch"):
        load_predictor(bundle_path)


def test_ml_predictor_requires_sklearn_compatible_predict(tmp_path):
    bundle_path = tmp_path / "no-predict.pkl"
    dump_bundle(bundle_path, model=object())

    with pytest.raises(TypeError, match="callable predict"):
        load_predictor(bundle_path)


def test_ml_predictor_requires_feature_metadata(tmp_path):
    bundle_path = tmp_path / "bare-model.pkl"
    joblib.dump(SklearnCompatibleRegressor(), bundle_path)

    with pytest.raises(ValueError, match="both 'model' and ordered 'features'"):
        load_predictor(bundle_path)
