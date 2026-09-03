import sys, torch
a = torch.load(sys.argv[1], weights_only=False); b = torch.load(sys.argv[2], weights_only=False)
L = a["layout"]; used = L["used"]; vs, ve = L["video_start"], L["video_start"] + L["num_frames"] * L["tokens_per_frame"]
def stats(x, y, name):
    x, y = x.float(), y.float()
    cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
    rel = ((x - y).norm() / y.norm()).item(); mx = (x - y).abs().max().item()
    print(f"{name:28s} cos={cos:.6f} relL2={rel:.3e} max|d|={mx:.3e} |ref|max={y.abs().max().item():.3e}")
for key in ("dense", "hybrid"):
    sg, vd = a[key][:used], b[key][:used]
    stats(sg, vd, f"{key} all rows")
    stats(sg[:L["text_len"]], vd[:L["text_len"]], f"{key} text rows")
    stats(sg[L["text_len"]:vs], vd[L["text_len"]:vs], f"{key} audio rows")
    stats(sg[vs:ve], vd[vs:ve], f"{key} video rows")
    tpf = L["tokens_per_frame"]
    stats(sg[vs:vs+tpf], vd[vs:vs+tpf], f"{key} video frame 0 (anchor)")
    stats(sg[vs+5*tpf:vs+6*tpf], vd[vs+5*tpf:vs+6*tpf], f"{key} video frame 5")
