"""Recipe opt-in for exact split-output extend paths; legacy disable switches win."""
import os

ENABLED = os.environ.get("SGLANG_Q38C_EXTEND_FASTPATH", "0") == "1"
MOE_ENABLED = ENABLED and os.environ.get(
    "SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_DISABLE", "false"
).lower() not in ("true", "1")
HC_ENABLED = ENABLED and os.environ.get(
    "SGLANG_X_HC_EXTEND_DISABLE", "0"
) not in ("1", "true", "True")
_GDN_ENABLED = ENABLED and os.environ.get("SGLANG_X_GDN_FRONT_DISABLE", "0") != "1"
GDN_CONV_ENABLED = _GDN_ENABLED and os.environ.get("SGLANG_X_GDN_FRONT_CONV_DISABLE", "0") != "1"
GDN_L2NORM_ENABLED = _GDN_ENABLED and os.environ.get("SGLANG_X_GDN_FRONT_L2NORM_DISABLE", "0") != "1"
GDN_NORM_ENABLED = _GDN_ENABLED and os.environ.get("SGLANG_X_GDN_FRONT_NORM_DISABLE", "0") != "1"
GDN_GATING_ENABLED = _GDN_ENABLED and os.environ.get("SGLANG_X_GDN_FRONT_GATING_DISABLE", "0") != "1"
