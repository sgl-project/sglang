"""Native processors with a metadata-only staged vision layout."""

FRAME_VIDEO_MODEL_TYPES = frozenset(
    {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "intern_s2_preview"}
)
TIMED_VIDEO_MODEL_TYPES = frozenset({"qwen2_5_vl", "qwen3_omni_moe"})
STAGED_VIDEO_MODEL_TYPES = (
    FRAME_VIDEO_MODEL_TYPES | TIMED_VIDEO_MODEL_TYPES | {"qwen2_vl"}
)
GLM_VIDEO_MODEL_TYPES = frozenset({"glm4v", "glm4v_moe", "glm5_next"})
NATIVE_VIDEO_MODEL_TYPES = GLM_VIDEO_MODEL_TYPES | {
    "muse_spark",
    "minimax_m3_vl",
    "mimo_v2",
}
NATIVE_IMAGE_MODEL_TYPES = NATIVE_VIDEO_MODEL_TYPES | {
    "dots3_note",
    "kimi_vl",
    "kimi_k25",
    "kimi_k3",
    "kimi_k31",
    "inkling2",
}
