import json

import pytest

from sglang.srt.arg_groups.speculative_hook import _resolve_speculative_algorithm_alias


@pytest.fixture
def qwen3_dspark_draft_path(tmp_path):
    draft_path = tmp_path / "draft"
    draft_path.mkdir()
    cfg = {
        "architectures": ["Qwen3DSparkDraftModel"],
        "block_size": 7,
        "hidden_size": 4096,
        "vocab_size": 151936,
    }
    (draft_path / "config.json").write_text(json.dumps(cfg))
    return str(draft_path)


def test_dspark_alias_routes_qwen3_draft_to_dflash(qwen3_dspark_draft_path):
    assert (
        _resolve_speculative_algorithm_alias("DSPARK", qwen3_dspark_draft_path)
        == "DFLASH"
    )
    assert (
        _resolve_speculative_algorithm_alias("DFLASH", qwen3_dspark_draft_path)
        == "DFLASH"
    )


def test_dspark_without_qwen3_draft_stays_dspark():
    assert _resolve_speculative_algorithm_alias("DSPARK", None) == "DSPARK"
