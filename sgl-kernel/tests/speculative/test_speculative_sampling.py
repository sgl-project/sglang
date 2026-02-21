import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import (
    tree_speculative_sampling_target_only,
    tree_speculative_sampling_target_only_rejmask,
)

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for sgl-kernel speculative sampling tests.",
        allow_module_level=True,
    )


def _build_kary_tree_structure(num_nodes: int, topk: int, device: str):
    """
    Build a k-ary tree in BFS order using two linked-list arrays:
      - next_token: parent -> first child (or -1)
      - next_sibling: child -> next sibling (or -1)

    Node 0 is treated as the root.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if not (1 <= topk <= 4):
        raise ValueError(f"topk must be in [1, 4], got {topk}")

    next_token = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)

    queue: list[int] = [0]
    next_free = 1
    while queue and next_free < num_nodes:
        parent = queue.pop(0)
        remaining = num_nodes - next_free
        num_children = min(topk, remaining)
        if num_children <= 0:
            continue

        first_child = next_free
        next_token[parent] = first_child
        prev_child = -1
        for _ in range(num_children):
            child = next_free
            if prev_child != -1:
                next_sibling[prev_child] = child
            prev_child = child
            queue.append(child)
            next_free += 1

    return next_token.contiguous(), next_sibling.contiguous()


def _build_tree_inputs(
    *,
    bs: int,
    num_draft_tokens: int,
    topk: int,
    vocab_size: int,
    seed: int,
    logit_scale: float,
    device: str,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # candidates: [bs, n], token ids in [0, vocab_size)
    candidates = torch.randint(
        0, vocab_size, (bs, num_draft_tokens), dtype=torch.int64, device=device
    )
    # Root token id is unused by the kernels, but keep it valid/stable.
    candidates[:, 0] = 0

    # retrive_index: [bs, n] maps to a flat predicts buffer of length bs*n
    base = (
        torch.arange(bs, device=device, dtype=torch.int64) * num_draft_tokens
    ).unsqueeze(1)
    retrive_index = base + torch.arange(
        num_draft_tokens, device=device, dtype=torch.int64
    ).unsqueeze(0)

    next_token_1d, next_sibling_1d = _build_kary_tree_structure(
        num_draft_tokens, topk, device
    )
    retrive_next_token = next_token_1d.unsqueeze(0).repeat(bs, 1).contiguous()
    retrive_next_sibling = next_sibling_1d.unsqueeze(0).repeat(bs, 1).contiguous()

    # Uniform samples: [bs, n] and [bs]
    coins = torch.rand((bs, num_draft_tokens), dtype=torch.float32, device=device)
    coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

    # target_probs: [bs, n, vocab_size] float32 probabilities
    logits = torch.randn(
        (bs, num_draft_tokens, vocab_size), dtype=torch.float32, device=device
    )
    logits = logits * float(logit_scale)
    target_probs = F.softmax(logits, dim=-1)

    return (
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token,
        retrive_next_sibling,
        coins.contiguous(),
        coins_for_final_sampling.contiguous(),
        target_probs.contiguous(),
    )


def _build_tree_inputs_with_structure(
    *,
    bs: int,
    num_draft_tokens: int,
    vocab_size: int,
    seed: int,
    logit_scale: float,
    device: str,
    next_token_1d: torch.Tensor,
    next_sibling_1d: torch.Tensor,
):
    """
    Same as _build_tree_inputs, but uses caller-provided 1D tree structure tensors
    (next_token_1d / next_sibling_1d) for all batch elements.
    """
    if (
        next_token_1d.numel() != num_draft_tokens
        or next_sibling_1d.numel() != num_draft_tokens
    ):
        raise ValueError("Structure tensors must have length == num_draft_tokens.")
    if next_token_1d.dtype != torch.int64 or next_sibling_1d.dtype != torch.int64:
        raise ValueError("Structure tensors must be int64.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    candidates = torch.randint(
        0, vocab_size, (bs, num_draft_tokens), dtype=torch.int64, device=device
    )
    candidates[:, 0] = 0

    base = (
        torch.arange(bs, device=device, dtype=torch.int64) * num_draft_tokens
    ).unsqueeze(1)
    retrive_index = base + torch.arange(
        num_draft_tokens, device=device, dtype=torch.int64
    ).unsqueeze(0)

    retrive_next_token = (
        next_token_1d.to(device=device).unsqueeze(0).repeat(bs, 1).contiguous()
    )
    retrive_next_sibling = (
        next_sibling_1d.to(device=device).unsqueeze(0).repeat(bs, 1).contiguous()
    )

    coins = torch.rand((bs, num_draft_tokens), dtype=torch.float32, device=device)
    coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

    logits = torch.randn(
        (bs, num_draft_tokens, vocab_size), dtype=torch.float32, device=device
    )
    logits = logits * float(logit_scale)
    target_probs = F.softmax(logits, dim=-1)

    return (
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token,
        retrive_next_sibling,
        coins.contiguous(),
        coins_for_final_sampling.contiguous(),
        target_probs.contiguous(),
    )


def _build_random_tree_structure(num_nodes: int, topk: int, seed: int, device: str):
    """
    Build a random rooted tree with max out-degree <= topk.
    Output format matches the kernels: next_token / next_sibling "linked lists".
    Nodes are labeled [0..num_nodes-1], with 0 as root.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if not (1 <= topk <= 4):
        raise ValueError(f"topk must be in [1, 4], got {topk}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    out_deg = [0 for _ in range(num_nodes)]

    # For each node i (except root), pick a parent among [0..i-1] that still has capacity.
    for i in range(1, num_nodes):
        parent = None
        # Try a few random picks first.
        for _ in range(16):
            cand = int(torch.randint(0, i, (1,), generator=g).item())
            if out_deg[cand] < topk:
                parent = cand
                break
        # Fallback: first available parent.
        if parent is None:
            for cand in range(i):
                if out_deg[cand] < topk:
                    parent = cand
                    break
        if parent is None:
            raise RuntimeError("Failed to assign parent under out-degree constraint.")

        children[parent].append(i)
        out_deg[parent] += 1

    # Shuffle sibling order for each parent to exercise next_sibling linkage.
    for p in range(num_nodes):
        if len(children[p]) > 1:
            perm = torch.randperm(len(children[p]), generator=g).tolist()
            children[p] = [children[p][j] for j in perm]

    next_token = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    for p in range(num_nodes):
        if not children[p]:
            continue
        next_token[p] = children[p][0]
        for a, b in zip(children[p], children[p][1:]):
            next_sibling[a] = b

    return next_token.contiguous(), next_sibling.contiguous()


def _build_star_tree_structure(num_nodes: int, topk: int, device: str):
    """
    Build a "star" tree: root has up to topk children, and all other nodes are leaves.
    Useful to exercise next_sibling linkage heavily while keeping depth minimal.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if not (1 <= topk <= 4):
        raise ValueError(f"topk must be in [1, 4], got {topk}")

    next_token = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    if num_nodes == 1:
        return next_token.contiguous(), next_sibling.contiguous()

    num_children = min(topk, num_nodes - 1)
    first_child = 1
    next_token[0] = first_child
    for i in range(first_child, first_child + num_children - 1):
        next_sibling[i] = i + 1
    return next_token.contiguous(), next_sibling.contiguous()


def _alloc_tree_outputs(*, bs: int, num_draft_tokens: int, device: str):
    # allow at most n-1 draft tokens accepted; keeping it n is safe and simplifies tests
    num_spec_tokens = num_draft_tokens
    predicts = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("bs,num_draft_tokens,vocab_size", [(1, 8, 16), (4, 16, 64)])
@pytest.mark.parametrize(
    "threshold_single,threshold_acc",
    [(1.0, 1.0), (0.0, 0.0), (0.9, 0.7)],
)
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("logit_scale", [0.0, 4.0])
@pytest.mark.parametrize("seed", [0, 7])
def test_tree_target_only_rejmask_matches_tree_topk_le_4(
    topk,
    bs,
    num_draft_tokens,
    vocab_size,
    threshold_single,
    threshold_acc,
    deterministic,
    logit_scale,
    seed,
):
    """
    Verify that when the tree branching factor (topk) is <= 4, the two kernel
    implementations produce identical outputs:
      - tree_speculative_sampling_target_only
      - tree_speculative_sampling_target_only_rejmask
    """
    device = "cuda"

    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        coins,
        coins_for_final_sampling,
        target_probs,
    ) = _build_tree_inputs(
        bs=bs,
        num_draft_tokens=num_draft_tokens,
        topk=topk,
        vocab_size=vocab_size,
        seed=seed,
        logit_scale=logit_scale,
        device=device,
    )

    predicts0, accept_index0, accept_token_num0 = _alloc_tree_outputs(
        bs=bs, num_draft_tokens=num_draft_tokens, device=device
    )
    predicts1, accept_index1, accept_token_num1 = _alloc_tree_outputs(
        bs=bs, num_draft_tokens=num_draft_tokens, device=device
    )

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=float(threshold_single),
        threshold_acc=float(threshold_acc),
        deterministic=bool(deterministic),
    )
    tree_speculative_sampling_target_only_rejmask(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=float(threshold_single),
        threshold_acc=float(threshold_acc),
        deterministic=bool(deterministic),
    )

    assert torch.equal(
        predicts0, predicts1
    ), "predicts mismatch between tree and tree_rejmask"
    assert torch.equal(
        accept_index0, accept_index1
    ), "accept_index mismatch between tree and tree_rejmask"
    assert torch.equal(
        accept_token_num0, accept_token_num1
    ), "accept_token_num mismatch between tree and tree_rejmask"


def test_tree_target_only_rejmask_matches_tree_chain_onehot_deterministic():
    """
    Additional coverage: a small, deterministic chain (topk=1) with nearly one-hot target
    distributions to make the expected path stable across runs.
    """
    device = "cuda"
    bs = 2
    num_draft_tokens = 4
    vocab_size = 16
    num_spec_tokens = 3

    # topk=1 chain: 0 -> 1 -> 2 -> 3, no siblings
    candidates = torch.tensor(
        [
            [0, 3, 5, 7],
            [0, 4, 6, 8],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, 3, -1],
            [1, 2, 3, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Nearly one-hot distributions so randomness doesn't affect results.
    target_logits = torch.full(
        (bs, num_draft_tokens, vocab_size), -10, dtype=torch.float32, device=device
    )
    target_logits[0, 0, candidates[0, 1].item()] = 10
    target_logits[1, 0, candidates[1, 1].item()] = 10
    target_logits[0, 1, candidates[0, 2].item()] = 10
    target_logits[1, 1, candidates[1, 2].item()] = 10
    target_logits[0, 2, candidates[0, 3].item()] = 10
    target_logits[1, 2, candidates[1, 3].item()] = 10

    target_probs = F.softmax(target_logits, dim=-1)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device, dtype=torch.float32)

    predicts0 = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index0 = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num0 = torch.full((bs,), 0, dtype=torch.int32, device=device)

    predicts1 = predicts0.clone()
    accept_index1 = accept_index0.clone()
    accept_token_num1 = accept_token_num0.clone()

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    tree_speculative_sampling_target_only_rejmask(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert torch.equal(predicts0, predicts1)
    assert torch.equal(accept_index0, accept_index1)
    assert torch.equal(accept_token_num0, accept_token_num1)


@pytest.mark.parametrize(
    "case",
    [
        # Fill the missing middle ground (topk=2/3 + medium shape).
        dict(
            topk=2,
            bs=2,
            num_draft_tokens=12,
            vocab_size=32,
            threshold_single=0.5,
            threshold_acc=1.0,
            deterministic=True,
            logit_scale=1.0,
            seed=1,
        ),
        dict(
            topk=2,
            bs=2,
            num_draft_tokens=12,
            vocab_size=32,
            threshold_single=0.9,
            threshold_acc=0.7,
            deterministic=False,
            logit_scale=4.0,
            seed=7,
        ),
        dict(
            topk=3,
            bs=2,
            num_draft_tokens=12,
            vocab_size=32,
            threshold_single=0.0,
            threshold_acc=0.0,
            deterministic=True,
            logit_scale=0.0,
            seed=0,
        ),
        dict(
            topk=3,
            bs=2,
            num_draft_tokens=12,
            vocab_size=32,
            threshold_single=0.9,
            threshold_acc=0.7,
            deterministic=False,
            logit_scale=10.0,
            seed=1,
        ),
        # Star-shaped trees stress next_sibling traversal (still topk<=4).
        dict(
            topk=4,
            bs=1,
            num_draft_tokens=9,
            vocab_size=31,
            threshold_single=0.5,
            threshold_acc=1.0,
            deterministic=True,
            logit_scale=4.0,
            seed=0,
            structure="star",
        ),
        dict(
            topk=4,
            bs=2,
            num_draft_tokens=9,
            vocab_size=31,
            threshold_single=0.0,
            threshold_acc=0.0,
            deterministic=False,
            logit_scale=10.0,
            seed=7,
            structure="star",
        ),
    ],
)
def test_tree_target_only_rejmask_matches_tree_selected_cases(case):
    """
    Targeted, non-explosive coverage to fill gaps not covered by the main parametrized test:
      - topk=2/3 medium shapes
      - star-shaped trees
      - threshold edge cases and different logit scales
    """
    device = "cuda"
    topk = int(case["topk"])
    bs = int(case["bs"])
    num_draft_tokens = int(case["num_draft_tokens"])
    vocab_size = int(case["vocab_size"])
    threshold_single = float(case["threshold_single"])
    threshold_acc = float(case["threshold_acc"])
    deterministic = bool(case["deterministic"])
    logit_scale = float(case["logit_scale"])
    seed = int(case["seed"])

    structure = case.get("structure", "kary")
    if structure == "star":
        next_token_1d, next_sibling_1d = _build_star_tree_structure(
            num_draft_tokens, topk, device
        )
        (
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            coins,
            coins_for_final_sampling,
            target_probs,
        ) = _build_tree_inputs_with_structure(
            bs=bs,
            num_draft_tokens=num_draft_tokens,
            vocab_size=vocab_size,
            seed=seed,
            logit_scale=logit_scale,
            device=device,
            next_token_1d=next_token_1d,
            next_sibling_1d=next_sibling_1d,
        )
    else:
        (
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            coins,
            coins_for_final_sampling,
            target_probs,
        ) = _build_tree_inputs(
            bs=bs,
            num_draft_tokens=num_draft_tokens,
            topk=topk,
            vocab_size=vocab_size,
            seed=seed,
            logit_scale=logit_scale,
            device=device,
        )

    predicts0, accept_index0, accept_token_num0 = _alloc_tree_outputs(
        bs=bs, num_draft_tokens=num_draft_tokens, device=device
    )
    predicts1, accept_index1, accept_token_num1 = _alloc_tree_outputs(
        bs=bs, num_draft_tokens=num_draft_tokens, device=device
    )

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )
    tree_speculative_sampling_target_only_rejmask(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )

    assert torch.equal(predicts0, predicts1)
    assert torch.equal(accept_index0, accept_index1)
    assert torch.equal(accept_token_num0, accept_token_num1)


@pytest.mark.parametrize("topk", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [0, 7])
def test_tree_target_only_rejmask_matches_tree_random_shapes_topk_le_4(topk, seed):
    """
    Additional coverage: compare kernels on random, irregular tree shapes (max out-degree <= topk),
    including small num_draft_tokens edge cases.
    """
    device = "cuda"

    sizes = [
        # Edge cases
        (1, 1, 7),  # root only
        (1, 2, 7),  # root + 1 node
        (1, 3, 11),  # tiny
        # Medium
        (2, 9, 31),
    ]
    thresholds = [
        (1.0, 1.0),
        (0.0, 0.0),
        (0.9, 0.7),
    ]
    det_flags = [True, False]
    logit_scales = [0.0, 10.0]

    for bs, num_draft_tokens, vocab_size in sizes:
        next_token_1d, next_sibling_1d = _build_random_tree_structure(
            num_draft_tokens, topk, seed=seed + num_draft_tokens * 13, device=device
        )
        for threshold_single, threshold_acc in thresholds:
            for deterministic in det_flags:
                for logit_scale in logit_scales:
                    (
                        candidates,
                        retrive_index,
                        retrive_next_token,
                        retrive_next_sibling,
                        coins,
                        coins_for_final_sampling,
                        target_probs,
                    ) = _build_tree_inputs_with_structure(
                        bs=bs,
                        num_draft_tokens=num_draft_tokens,
                        vocab_size=vocab_size,
                        seed=seed,
                        logit_scale=logit_scale,
                        device=device,
                        next_token_1d=next_token_1d,
                        next_sibling_1d=next_sibling_1d,
                    )

                    predicts0, accept_index0, accept_token_num0 = _alloc_tree_outputs(
                        bs=bs, num_draft_tokens=num_draft_tokens, device=device
                    )
                    predicts1, accept_index1, accept_token_num1 = _alloc_tree_outputs(
                        bs=bs, num_draft_tokens=num_draft_tokens, device=device
                    )

                    tree_speculative_sampling_target_only(
                        predicts=predicts0,
                        accept_index=accept_index0,
                        accept_token_num=accept_token_num0,
                        candidates=candidates,
                        retrive_index=retrive_index,
                        retrive_next_token=retrive_next_token,
                        retrive_next_sibling=retrive_next_sibling,
                        uniform_samples=coins,
                        uniform_samples_for_final_sampling=coins_for_final_sampling,
                        target_probs=target_probs,
                        threshold_single=float(threshold_single),
                        threshold_acc=float(threshold_acc),
                        deterministic=bool(deterministic),
                    )
                    tree_speculative_sampling_target_only_rejmask(
                        predicts=predicts1,
                        accept_index=accept_index1,
                        accept_token_num=accept_token_num1,
                        candidates=candidates,
                        retrive_index=retrive_index,
                        retrive_next_token=retrive_next_token,
                        retrive_next_sibling=retrive_next_sibling,
                        uniform_samples=coins,
                        uniform_samples_for_final_sampling=coins_for_final_sampling,
                        target_probs=target_probs,
                        threshold_single=float(threshold_single),
                        threshold_acc=float(threshold_acc),
                        deterministic=bool(deterministic),
                    )

                    assert torch.equal(
                        predicts0, predicts1
                    ), "predicts mismatch between tree and tree_rejmask (random tree)"
                    assert torch.equal(
                        accept_index0, accept_index1
                    ), "accept_index mismatch between tree and tree_rejmask (random tree)"
                    assert torch.equal(
                        accept_token_num0, accept_token_num1
                    ), "accept_token_num mismatch between tree and tree_rejmask (random tree)"


def test_tree_speculative_sampling_target_only_rejmask_reject_masks_token_in_final_sampling():
    """
    Construct coins to force a rejection at a specific step, then verify the final sampling
    will not sample the rejected token (mask == relu(q - p) equivalence for topk==1).
    """
    device = "cuda"
    bs = 1
    num_draft_tokens = 3
    vocab_size = 32
    num_spec_tokens = 3

    t1 = 3  # token at node 1 (will be accepted)
    t2 = 5  # token at node 2 (will be rejected)
    alt = 31  # token that should be sampled after masking t2
    candidates = torch.tensor([[0, t1, t2]], dtype=torch.int64, device=device)
    retrive_index = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    retrive_next_token = torch.tensor([[1, 2, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    # Build target_probs directly (no softmax) for full control:
    # - At root (slot 0): P(t1)=0.95 => accept by threshold_single.
    # - At node1 (slot 1): P(t2)=0.90, P(alt)=0.10. We will reject t2 (by threshold + coin), then final sampling
    #   must NOT return t2; it should return alt given the chosen coin.
    target_probs = torch.zeros(
        (bs, num_draft_tokens, vocab_size), dtype=torch.float32, device=device
    )
    target_probs[0, 0, 0] = 0.05
    target_probs[0, 0, t1] = 0.95
    target_probs[0, 1, t2] = 0.90
    target_probs[0, 1, alt] = 0.10
    target_probs[0, 2, 0] = 1.0

    # Coins:
    # - First decision uses uniform_samples[0,0] but accept is forced by threshold_single.
    # - Second decision (for node2) uses uniform_samples[0,1] and must reject:
    #   coin=0.99, prob_acc=P(t2)=0.90, threshold_acc=1.0 => coin > prob_acc, and threshold_single > 0.90 => reject.
    coins = torch.tensor([[0.0, 0.99, 0.0]], dtype=torch.float32, device=device)
    # Final sampling coin: coin=0.5
    # - Without masking: u=0.5*1.0=0.5 and since rejected token (id=5) appears before alt (id=31)
    #   with prob 0.9, it would be sampled.
    # - With masking: sum=0.1, u=0.05, only alt has mass => alt is sampled.
    coins_for_final_sampling = torch.tensor([0.5], dtype=torch.float32, device=device)

    predicts0 = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index0 = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num0 = torch.full((bs,), 0, dtype=torch.int32, device=device)

    predicts1 = predicts0.clone()
    accept_index1 = accept_index0.clone()
    accept_token_num1 = accept_token_num0.clone()

    # Make sure t1 is accepted (0.95 >= 0.91) but t2 is NOT (0.90 < 0.91).
    threshold_single = 0.91
    threshold_acc = 1.0

    tree_speculative_sampling_target_only(
        predicts=predicts0,
        accept_index=accept_index0,
        accept_token_num=accept_token_num0,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    tree_speculative_sampling_target_only_rejmask(
        predicts=predicts1,
        accept_index=accept_index1,
        accept_token_num=accept_token_num1,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert torch.equal(predicts0, predicts1)
    assert torch.equal(accept_index0, accept_index1)
    assert torch.equal(accept_token_num0, accept_token_num1)

    # Node 1 token is accepted and written to predicts[0]
    assert predicts0[0].item() == t1
    # Final sampled token is written at last_accepted_retrive_idx (node 1's retrive_index == 1)
    final_token = predicts0[1].item()
    assert final_token != t2
    assert final_token == alt


if __name__ == "__main__":
    pytest.main([__file__])
