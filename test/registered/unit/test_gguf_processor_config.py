from types import SimpleNamespace

from sglang.srt.utils.hf_transformers import processor as processor_module


def test_processor_resolves_local_gguf_with_gguf_aware_config(
    tmp_path, monkeypatch
):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()

    config_calls = []

    def fake_get_config(model, **kwargs):
        config_calls.append((model, kwargs))
        return SimpleNamespace(model_type="test")

    def fail_auto_config(*args, **kwargs):
        raise AssertionError("AutoConfig must not parse a local GGUF as JSON")

    class FakeTokenizer:
        chat_template = "template"

        def get_added_vocab(self):
            return {}

    fake_processor = SimpleNamespace(tokenizer=FakeTokenizer())

    monkeypatch.setattr(processor_module, "get_config", fake_get_config)
    monkeypatch.setattr(
        processor_module.AutoConfig, "from_pretrained", fail_auto_config
    )
    monkeypatch.setattr(
        processor_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: fake_processor,
    )
    monkeypatch.setattr(
        processor_module, "patch_mistral_common_tokenizer", lambda tokenizer: None
    )
    monkeypatch.setattr(
        processor_module, "_fix_special_tokens_pattern", lambda tokenizer: None
    )
    monkeypatch.setattr(
        processor_module, "_fix_added_tokens_encoding", lambda tokenizer: None
    )
    monkeypatch.setattr(
        processor_module, "attach_additional_stop_token_ids", lambda tokenizer: None
    )

    result = processor_module.get_processor(
        str(tokenizer_path),
        model_name=str(gguf_path),
        trust_remote_code=True,
        revision="revision",
    )

    assert result is fake_processor
    assert config_calls == [
        (
            str(gguf_path),
            {"trust_remote_code": True, "revision": "revision"},
        )
    ]
