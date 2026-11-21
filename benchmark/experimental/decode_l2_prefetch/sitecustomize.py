import os

if os.environ.get("SGLANG_DECODE_L2PF", "0") != "0":
    import decode_l2pf

    decode_l2pf.install()
