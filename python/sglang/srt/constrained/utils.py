from typing import Dict


def is_legacy_structural_tag(obj: Dict) -> bool:
    # test whether an object is a legacy structural tag
    # see `StructuralTagResponseFormat` at `sglang.srt.entrypoints.openai.protocol`
    if obj.get("structures", None) is not None:
        assert obj.get("triggers", None) is not None
        return True
    else:
        assert obj.get("format", None) is not None
        return False


def structural_tag_format(obj: Dict) -> Dict:
    """The format of a structural tag, converting a legacy tag the way
    xgrammar's StructuralTag.from_legacy_structural_tag does."""
    if not is_legacy_structural_tag(obj):
        return obj["format"]
    return {
        "type": "triggered_tags",
        "triggers": obj["triggers"],
        "tags": [
            {
                "type": "tag",
                "begin": structure["begin"],
                "content": {"type": "json_schema", "json_schema": structure["schema"]},
                "end": structure["end"],
            }
            for structure in obj["structures"]
        ],
        "at_least_one": obj.get("at_least_one", False),
    }
