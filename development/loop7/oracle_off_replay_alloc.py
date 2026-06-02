"""Loop-7 AC-1 evidence: with the recall oracle OFF (default), the production
graph-safe DS decode selector under CUDA-graph capture/replay yields
byte-identical selected_indices/valid_lengths vs the eager path AND does zero new
CUDA allocations under replay ("zero hot-path cost" — demonstrated, not asserted).

Emits oracle_off_graph_replay_alloc.json. GPU-only.
"""
import hashlib
import json

import torch

from sglang.srt.layers.attention.double_sparsity.channel_mask import ChannelMask
from sglang.srt.layers.attention.double_sparsity.config import parse_double_sparsity_config
from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
    allocate_graph_state, assert_no_alloc_in_region, capture_decode_step,
)
from sglang.srt.layers.attention.double_sparsity.selector import DoubleSparsitySelector
from sglang.srt.layers.attention.double_sparsity.token_label_table import (
    allocate_token_label_table,
)

OUT = "development/loop7/oracle_off_graph_replay_alloc.json"
REPLAY_STEPS = 120


def _sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    assert torch.cuda.is_available(), "GPU required"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    L, T, H, Ld, hd, K, bs, msl = 1, 64, 4, 8, 8, 8, 2, 48
    cfg = parse_double_sparsity_config(
        '{"top_k": %d, "page_size": 64, "channel_mask_path": "/tmp/x.safetensors",'
        ' "device_buffer_size": 4096}' % K
    )
    assert cfg.recall_oracle is False  # oracle OFF (default)
    sel = DoubleSparsitySelector(config=cfg, num_local_heads=H, head_dim=hd, device=dev)
    table = allocate_token_label_table(
        num_layers_local=L, max_tokens=T, num_heads_local=H, label_dim=Ld,
        page_size=64, dtype=torch.float32, device=dev,
    )
    table.signatures[0] = torch.randn(T, H, Ld, device=dev)
    table.written[0, :] = True
    mask = ChannelMask(
        channel_selection=torch.arange(Ld, device=dev).view(1, 1, -1).expand(L, H, -1).to(torch.int32).contiguous(),
        channel_weights=torch.ones(L, H, Ld, dtype=torch.float32, device=dev),
        schema_version="1", dtype="fp8_e4m3", head_dim=hd, page_size=64,
        label_dim=Ld, content_sha256="test",
    )
    sel.bind_runtime_data(table, mask)

    queries = torch.randn(bs, H, hd, device=dev)
    req_pool = torch.arange(bs, dtype=torch.int32, device=dev)
    sparse_mask = torch.ones(bs, msl, dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([msl, msl - 7], dtype=torch.int32, device=dev)
    req_to_token = torch.arange(msl, dtype=torch.int32, device=dev).unsqueeze(0).expand(bs, -1).contiguous()

    # Eager baseline (oracle off) — the reference selection.
    idx_e, len_e = sel.retrieve_topk(
        queries=queries, layer_id=0, req_pool_indices=req_pool,
        sparse_mask=sparse_mask, seq_lens=seq_lens, req_to_token=req_to_token,
    )
    torch.cuda.synchronize()

    state = allocate_graph_state(max_bs=bs, max_top_k=K, max_seq_len=msl,
                                 num_local_heads=H, label_dim=Ld, device=dev)
    replay = capture_decode_step(
        sel, state=state, queries=queries, layer_id=0, req_pool_indices=req_pool,
        sparse_mask=sparse_mask, seq_lens=seq_lens, req_to_token=req_to_token,
        max_seq_len=msl,
    )
    torch.cuda.synchronize()

    # Replay many steps; every step must equal the eager baseline.
    all_idx_equal = True
    all_len_equal = True
    for _ in range(REPLAY_STEPS):
        idx_r, len_r = replay()
        torch.cuda.synchronize()
        all_idx_equal &= bool(torch.equal(idx_r[:bs, :K], idx_e))
        all_len_equal &= bool(torch.equal(len_r[:bs], len_e))

    # Zero-allocation probe: one more replay inside the alloc detector.
    before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    alloc_violation = False
    try:
        with assert_no_alloc_in_region("oracle-off-replay"):
            replay()
            torch.cuda.synchronize()
    except RuntimeError:
        alloc_violation = True
    after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    alloc_delta = after - before

    verdict = bool(all_idx_equal and all_len_equal and not alloc_violation and alloc_delta == 0)
    out = {
        "what": "oracle-off graph-safe DS selector: byte-identical vs eager + zero-alloc under CUDA-graph replay",
        "recall_oracle": cfg.recall_oracle,
        "graph_mode": "cuda_graph_capture_replay",
        "dtype": "fp32_signatures",
        "top_k": K, "bs": bs, "max_seq_len": msl, "num_heads": H, "label_dim": Ld,
        "replay_steps": REPLAY_STEPS,
        "eager_indices_sha16": _sha(idx_e),
        "replay_indices_sha16": _sha(idx_r[:bs, :K]),
        "eager_lengths_sha16": _sha(len_e),
        "replay_indices_byte_identical_to_eager": all_idx_equal,
        "replay_lengths_byte_identical_to_eager": all_len_equal,
        "replay_allocation_delta_bytes": int(alloc_delta),
        "replay_zero_new_allocations": (not alloc_violation and alloc_delta == 0),
        "verdict": "PASS" if verdict else "FAIL",
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote -> {OUT}")
    if not verdict:
        raise SystemExit("AC-1 oracle-off replay/alloc check FAILED")


if __name__ == "__main__":
    main()
