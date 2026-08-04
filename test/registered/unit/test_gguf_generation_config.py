from pathlib import Path

from sglang.srt.utils.hf_transformers import common


def test_generation_config_uses_adjacent_directory_for_local_gguf(
    tmp_path, monkeypatch
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    calls = []
    sentinel = object()

    class FakeGenerationConfig:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return sentinel

    monkeypatch.setattr(common, "GenerationConfig", FakeGenerationConfig)

    result = common.get_generation_config(
        str(gguf_path),
        trust_remote_code=True,
        revision="revision",
    )

    assert result is sentinel
    assert calls == [
        (
            str(Path(gguf_path).parent),
            {"trust_remote_code": True, "revision": "revision"},
        )
    ]
