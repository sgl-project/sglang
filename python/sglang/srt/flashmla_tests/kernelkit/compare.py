from typing import List

import torch


def check_is_bitwise_equal_comparator(
    ans: torch.Tensor, ref: torch.Tensor, result: torch.Tensor
):
    """
    Return if two tensors are bitwise equal
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert ans.shape == ref.shape, "Shape mismatch"
    torch.all(torch.eq(ans, ref), out=result)


def check_is_bitwise_equal(
    name: str, ans: torch.Tensor, ref: torch.Tensor, quiet: bool = False
) -> bool:
    is_bitwise_equal = torch.equal(ans, ref)
    if not quiet and not is_bitwise_equal:
        print(
            f"`{name}` mismatch: not bitwise equal. Mismatch count: {(ans != ref).sum().item()} out of {ans.numel()}"
        )
    return is_bitwise_equal


def get_cos_diff(ans: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Calculate the cosine diff between two tensors
    Return a float if avoid_sync is False, else return a tensor
    """
    ans, ref = ans.double(), ref.double()
    if (ref * ref).sum().item() < 1e-12:
        return 0
    denominator = (ans * ans + ref * ref).sum().item()
    sim = 2 * (ans * ref).sum().item() / denominator
    return 1 - sim


def check_is_allclose(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
    quiet: bool = False,
) -> bool:
    """
    Check if two tensors are close enough
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert (
        ans.shape == ref.shape
    ), f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    assert (
        ans.dtype == ref.dtype
    ), f"`{name}` Dtype mismatch: {ans.dtype} vs {ref.dtype}"

    ans = ans.clone().to(torch.float)
    ref = ref.clone().to(torch.float)

    def report_err(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    # Deal with anomalies
    def deal_with_anomalies(val: float):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            report_err(
                f"`{name}` Anomaly number `{val}` mismatch: {ans_mask.sum().item()} in ans but {ref_mask.sum().item()} in ref"
            )
            return False
        return True

    anomalies_check_passed = True
    anomalies_check_passed &= deal_with_anomalies(float("inf"))
    anomalies_check_passed &= deal_with_anomalies(float("-inf"))
    anomalies_check_passed &= deal_with_anomalies(float("nan"))

    cos_diff = get_cos_diff(ans, ref)
    raw_abs_err = torch.abs(ans - ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref) + (1e-6))
    rel_err = raw_rel_err.masked_fill(raw_abs_err < abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err < rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not anomalies_check_passed:
        return False

    if not pass_mask.all():
        report_err(f"`{name}` mismatch")
        max_abs_err_pos: int = torch.argmax(abs_err, keepdim=True).item()
        max_rel_err_pos: int = torch.argmax(rel_err, keepdim=True).item()

        def get_pos_in_tensor(t: torch.Tensor, pos: int) -> List[int]:
            result = []
            for size in t.shape[::-1]:
                result.append(pos % size)
                pos = pos // size
            assert pos == 0
            return result[::-1]

        report_err(
            f"max abs err: {torch.max(abs_err).item()}: pos {get_pos_in_tensor(ans, max_abs_err_pos)}, {ans.reshape(-1)[max_abs_err_pos].item()} vs {ref.reshape(-1)[max_abs_err_pos].item()}"
        )
        report_err(
            f"max rel err: {torch.max(rel_err).item()}: pos {get_pos_in_tensor(ans, max_rel_err_pos)}, {ans.reshape(-1)[max_rel_err_pos].item()} vs {ref.reshape(-1)[max_rel_err_pos].item()}"
        )
        report_err(
            f"{pass_mask.sum()} out of {pass_mask.numel()} passed ({pass_mask.sum()/pass_mask.numel()*100.0:.2f}%)"
        )
        report_err(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    else:
        if abs(cos_diff) > cos_diff_tol:
            report_err(
                f"`{name}` mismatch: Cosine diff too large: {cos_diff} vs {cos_diff_tol})"
            )
            return False
        return True


def check_is_allclose_comparator(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    out: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
):
    out.fill_(check_is_allclose(name, ans, ref, abs_tol, rel_tol, cos_diff_tol))
