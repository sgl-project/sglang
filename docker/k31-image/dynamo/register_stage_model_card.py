# K3.1 image: stage the K3.1 model card on first use, so Dynamo serves Kimi-K3.1 without a
# per-job setup script. The layout matches dynamo_sglang_k31_patch.py: a symlink to every
# non-weight file of the model plus a config.json whose model_type is "kimi_k3". A setup script
# that stages it first still works; this only fills in a missing or stale directory.
_k31_image_d002_register = _register_model_source_path


def _k31_image_stage_model_card(source: str) -> None:
    import shutil
    import tempfile

    try:
        with open(os.path.join(source, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(config, dict) or "KimiK31ForConditionalGeneration" not in (
        config.get("architectures") or []
    ):
        return
    staged = _k31_model_card_staged_dir()
    expected_config = {**config, "model_type": "kimi_k3"}
    names = sorted(
        name
        for name in os.listdir(source)
        if name != "config.json"
        and not _k31_model_card_is_weight_file(name)
        and os.path.isfile(os.path.join(source, name))
    )
    try:
        with open(os.path.join(staged, "config.json"), encoding="utf-8") as f:
            current = json.load(f)
        links = sorted(name for name in os.listdir(staged) if name != "config.json")
        if (
            current == expected_config
            and links == names
            and all(
                os.path.realpath(os.path.join(staged, name))
                == os.path.realpath(os.path.join(source, name))
                for name in names
            )
        ):
            return
    except (OSError, ValueError):
        pass
    os.makedirs(_K31_MODEL_CARD_STAGE_ROOT, mode=0o755, exist_ok=True)
    tmp = tempfile.mkdtemp(
        dir=_K31_MODEL_CARD_STAGE_ROOT, prefix=os.path.basename(staged) + ".tmp."
    )
    try:
        os.chmod(tmp, 0o755)
        for name in names:
            os.symlink(os.path.join(source, name), os.path.join(tmp, name))
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(expected_config, f, indent=2)
        if os.path.lexists(staged):
            old = tempfile.mkdtemp(
                dir=_K31_MODEL_CARD_STAGE_ROOT,
                prefix=os.path.basename(staged) + ".old.",
            )
            os.rename(staged, os.path.join(old, "d"))
            shutil.rmtree(old, ignore_errors=True)
        os.rename(tmp, staged)
    except OSError:
        # Another process staged it concurrently; the patched function validates the result.
        shutil.rmtree(tmp, ignore_errors=True)


def _register_model_source_path(engine, server_args):
    _k31_image_stage_model_card(
        _k31_model_card_original_source_path(engine, server_args)
    )
    return _k31_image_d002_register(engine, server_args)
