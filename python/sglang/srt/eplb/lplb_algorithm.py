import torch
from sglang_kernel import ipm_solve


def lplb_algorithm(
    phy2log: torch.Tensor,
    logcnt: torch.Tensor,
    log2phy: torch.Tensor,
    g: int,
    logical_count: torch.Tensor,
    device: torch.device,
):
    log2phy_prob = torch.zeros(log2phy.shape, dtype=torch.float32, device=device)
    for layer_id in range(phy2log.shape[0]):
        layer_phy2log = phy2log[layer_id]
        layer_logcnt = logcnt[layer_id]
        layer_log2phy = log2phy[layer_id]
        logical_count = logical_count[layer_id]
        log2phy_prob = single_layer_lplb_algorithm(
            layer_phy2log, layer_logcnt, layer_log2phy, g, logical_count, device
        )
        log2phy_prob[layer_id] = log2phy_prob
    return log2phy_prob


def single_layer_lplb_algorithm(
    layer_phy2log: torch.Tensor,
    layer_logcnt: torch.Tensor,
    layer_log2phy: torch.Tensor,
    g: int,
    logical_count: torch.Tensor,
    device: torch.device,
):
    layer_phy2log = layer_phy2log.to(device)
    layer_logcnt = layer_logcnt.to(device)
    layer_log2phy = layer_log2phy.to(device)
    logical_count = logical_count.to(device)

    num_phy: int = layer_phy2log.shape[0]
    num_phy_gpu: int = num_phy // g

    log_single_expert_array: torch.Tensor = torch.nonzero(layer_logcnt == 1).flatten()
    phy_single_expert_array: torch.Tensor = layer_log2phy[log_single_expert_array, 0]
    log_replicated_expert_array: torch.Tensor = torch.nonzero(
        layer_logcnt > 1
    ).flatten()
    phy_replicated_expert_array: torch.Tensor = torch.nonzero(
        layer_logcnt[layer_phy2log] > 1
    ).flatten()

    single_expert_count: int = len(log_single_expert_array)
    log_replicated_expert_count: int = len(log_replicated_expert_array)
    phy_replicated_expert_count: int = len(phy_replicated_expert_array)

    B = torch.zeros((g, num_phy), dtype=torch.float32, device=device)
    for i in range(g):
        B[i, i * num_phy_gpu : (i + 1) * num_phy_gpu] = 1
    B1 = B[:, phy_single_expert_array]
    B2 = B[:, phy_replicated_expert_array]

    # Create C matrix using torch operations
    C = torch.zeros(
        (log_replicated_expert_count, phy_replicated_expert_count),
        dtype=torch.float32,
        device=device,
    )
    phy2log_rep = layer_phy2log[phy_replicated_expert_array]
    for i in range(log_replicated_expert_count):
        C[i, phy2log_rep == log_replicated_expert_array[i]] = 1.0

    # Construct matrix A = [[C, 0, 0, 1000], [B2, I, -1, 1000]]
    zeros_top = torch.zeros(
        (log_replicated_expert_count, g), dtype=torch.float32, device=device
    )
    zeros_top_col = torch.zeros(
        (log_replicated_expert_count, 1), dtype=torch.float32, device=device
    )

    I_matrix = torch.eye(g, dtype=torch.float32, device=device)
    neg_ones_col = torch.full((g, 1), -1.0, dtype=torch.float32, device=device)

    # Construct the matrix using torch block operations
    A_top = torch.hstack([C, zeros_top, zeros_top_col])
    A_bottom = torch.hstack([B2, I_matrix, neg_ones_col])
    A = torch.vstack([A_top, A_bottom])

    c = torch.zeros(A.shape[1] + 1, dtype=torch.float32, device=device)
    c[-2] = 1.0
    c[-1] = 1000.0

    logical_count = logical_count.to(torch.float32)
    logical_count = logical_count / logical_count.sum()
    t1: torch.Tensor = logical_count[log_single_expert_array]
    left = B1 @ t1
    b2 = -left.flatten()
    b1 = logical_count[log_replicated_expert_array].to(torch.float32)
    b = torch.cat([b1, b2])

    big_M_col = b - torch.sum(A, dim=1)
    A = torch.hstack([A, big_M_col.reshape(-1, 1)])

    avail_counter = torch.zeros((), dtype=torch.int, device=device)

    result = ipm_solve(
        A,
        b,
        c,
        avail_counter,
        None,
    )

    x = result[: phy_replicated_expert_array.shape[0]]
    x[x < 0.0] = 0.0
    phy_prob = torch.zeros(
        log_single_expert_array.shape[0] + phy_replicated_expert_array.shape[0] + 1,
        dtype=torch.float32,
        device=device,
    )
    phy_prob[phy_replicated_expert_array] = x
    phy_prob[phy_single_expert_array] = t1
    log2phy_prob = torch.zeros(layer_log2phy.shape, dtype=torch.float32, device=device)
    log2phy_prob = torch.take(phy_prob, layer_log2phy)

    return log2phy_prob
