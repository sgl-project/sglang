from sglang.srt.utils import get_bool_env_var

NSA_USE_REAL_INDEXER = get_bool_env_var("SGLANG_NSA_USE_REAL_INDEXER", "true")
NSA_DUAL_STREAM = get_bool_env_var("SGLANG_NSA_DUAL_STREAM", "true")
NSA_FUSE_TOPK = get_bool_env_var("SGLANG_NSA_FUSE_TOPK", "true")


def _print_bool_env_vars():
    msg = ""
    for k, v in globals().items():
        if k.startswith("NSA_") and isinstance(v, bool):
            msg += f"{k}={v} "
    print(msg, flush=True)


_print_bool_env_vars()


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)
