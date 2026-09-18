import pytest
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


def test_parse_dflash_draft_config_raises_on_sample_from_anchor():
    config = {
        "dflash_config": {
            "sample_from_anchor": True,
        }
    }
    with pytest.raises(
        ValueError, match="sample_from_anchor=True is not supported for DFlash."
    ):
        parse_dflash_draft_config(draft_hf_config=config)


def test_parse_dflash_draft_config_raises_on_query_zero_predicts_next():
    config = {
        "dflash_config": {
            "query_zero_predicts_next": True,
        }
    }
    with pytest.raises(
        ValueError, match="query_zero_predicts_next=True is not supported for DFlash."
    ):
        parse_dflash_draft_config(draft_hf_config=config)
