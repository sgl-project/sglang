import json
import os
import pickle
import stat
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.transport.request import (
    RequestUnpickler,
    request_pickle_globals,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.layers.attention.STA_configuration import (
    read_specific_json_files,
    save_mask_search_results,
)


def test_search_ids_are_per_submission_not_per_prompt_or_worker():
    args = SimpleNamespace(attention_backend_config={}, enable_trace=False)
    params = SamplingParams(prompt="synthetic calibration", request_id="caller-label")
    assert "sta_search_artifact_id" not in prepare_request(args, params).extra
    args.attention_backend_config["STA_mode"] = "STA_SEARCHING"
    request = prepare_request(args, params)
    key = request.extra["sta_search_artifact_id"]
    assert UUID(key).version == 4
    restored = RequestUnpickler(
        pickle.dumps(request), request_pickle_globals(SamplingParams)
    ).load()
    assert restored.extra["sta_search_artifact_id"] == key
    assert prepare_request(args, params).extra["sta_search_artifact_id"] != key


def test_tuning_reads_new_and_legacy_losses_without_printing_legacy_names(
    tmp_path, capsys, monkeypatch
):
    key = "c018f54b99264cc29755f7310d380172"
    losses = [{"L1_loss": [0.25], "L2_loss": [0.125]}]
    expected = {"L1_loss": {"[1, 2, 3]": [[0.25]]}, "L2_loss": {"[1, 2, 3]": [[0.125]]}}
    for side in ("pos", "neg"):
        directory = tmp_path / side
        path = Path(save_mask_search_results(losses, key, ["1,2,3"], str(directory)))
        assert path == directory / f"mask_search_{key}.json"
        assert json.loads(path.read_text()) == expected
        if os.name == "posix":
            assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
        legacy = directory / "mask_search_legacy_synthetic_prompt.json"
        legacy.write_text(path.read_text())
        assert read_specific_json_files(str(directory)) == [expected, expected]
        assert legacy.name not in capsys.readouterr().out

        def interrupt(data, f, **kwargs):
            f.write("{")
            raise OSError("synthetic interruption")

        with monkeypatch.context() as m:
            m.setattr(json, "dump", interrupt)
            with pytest.raises(OSError, match="synthetic interruption"):
                save_mask_search_results(losses, key, ["1,2,3"], str(directory))
        assert json.loads(path.read_text()) == expected
        assert set(directory.iterdir()) == {path, legacy}
