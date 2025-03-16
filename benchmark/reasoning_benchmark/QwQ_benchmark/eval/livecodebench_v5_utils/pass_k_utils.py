import numpy as np


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(results):
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades
