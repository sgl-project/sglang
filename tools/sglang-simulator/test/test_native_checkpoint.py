"""Bind the actual native HF serializer to the SDK's unchanged source config."""

import pytest
from test_cpu_bootstrap import requires_cpu_runtime, run_python


@requires_cpu_runtime
def test_real_native_hf_checkpoint_mapping_and_tamper_refusal():
    pytest.importorskip("simopus")
    run_python(
        """
import copy, json, tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
install_simulator_hooks()
from simopus.sdk.models.model_config import load_model_config
from simopus.sdk.perf_database.dsv41_dspark_proposal import MODEL_REVISION
from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.utils.hf_transformers_patches import normalize_rope_scaling_compat
from sglang_simulator.time_predictor.native_checkpoint import _bind_local_model_config
from sglang_simulator.time_predictor.base import PredictorError
original = load_model_config('deepseek-ai/DeepSeek-V4.1-Flash', revision=MODEL_REVISION, allow_download=False)
native = DeepseekV41Config(**copy.deepcopy(original))
normalize_rope_scaling_compat(native)
with tempfile.TemporaryDirectory() as directory:
    source = Path(directory)/'config.json'
    source.write_text(json.dumps(original))
    model = NS(model_path=directory, hf_config=native)
    receipt = _bind_local_model_config(model, original)
    assert receipt['stable_binding'] is True
    assert source.read_text() == json.dumps(original)
    model.hf_config = native.to_dict()
    model.hf_config['hidden_size'] += 1
    try:
        _bind_local_model_config(model, original)
    except PredictorError as error:
        assert error.code == 'incompatible_runtime'
    else:
        raise AssertionError('native checkpoint geometry tamper was accepted')
""",
        bootstrap=True,
        cpu=True,
    )
