def patch_vllm_linear_base_isinstance():
    import builtins

    from vllm.model_executor.layers.linear import LinearBase

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase

    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


patch_vllm_linear_base_isinstance()
