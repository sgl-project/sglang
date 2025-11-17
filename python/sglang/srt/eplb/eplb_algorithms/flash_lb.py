import time
import math

import numpy as np
import torch

from numba import njit
from scipy import stats
from scipy.optimize import linear_sum_assignment


@njit
def min_max_replica(mu, var, num_available_replicas, current_replicas, z_score):
    N = mu.shape[0]
    unit_value = (mu + z_score * np.sqrt(var)) / current_replicas

    replicas_history = np.ones((num_available_replicas + 1, N), dtype=np.int32)
    replicas_history[0, :] = current_replicas[:]

    for r in range(num_available_replicas):
        max_idx = -1
        max_value = -1.0

        for idx in range(N):
            value = unit_value[idx]
            if value > max_value:
                max_value = value
                max_idx = idx

        current_replicas[max_idx] += 1
        unit_value[max_idx] = (mu[max_idx] + z_score * np.sqrt(var[max_idx])) / current_replicas[max_idx]

        replicas_history[r + 1, :] = current_replicas[:]

    return current_replicas, replicas_history


@njit(fastmath=True, cache=True)
def compute_updated_device_variance(new_expert_id, device_slots, current_device_var, expert_var, expert_cov,
                                    expert_replicas):
    new_device_var = current_device_var + expert_var[new_expert_id] / expert_replicas[new_expert_id] ** 2

    for slot in device_slots:
        if slot == -1:
            break
        new_device_var += 2 * expert_cov[new_expert_id, slot] / expert_replicas[new_expert_id] / expert_replicas[slot]

    return new_device_var


@njit(fastmath=True, cache=True)
def lpt_deployment(mu, var, cov, deployment, deployed_replicas, total_replicas, z_score):
    num_devices, num_slots_per_device = deployment.shape

    unit_value = mu / total_replicas
    sorted_indices = np.argsort(-unit_value)

    new_deployment = -np.ones_like(deployment)
    device_mu = np.zeros(num_devices, dtype=np.float32)
    device_var = np.zeros(num_devices, dtype=np.float32)
    dev_ptr = np.zeros(num_devices, dtype=np.int32)

    # 快速初始化
    for dev in range(num_devices):
        for slot in deployment[dev]:
            if slot != -1:
                device_mu[dev] += mu[slot] / total_replicas[slot]
                device_var[dev] += compute_updated_device_variance(slot, new_deployment[dev], device_var[dev], var, cov,
                                                                   total_replicas)
                new_deployment[dev, dev_ptr[dev]] = slot
                dev_ptr[dev] += 1

    for idx in sorted_indices:
        for _ in range(total_replicas[idx] - deployed_replicas[idx]):
            best_dev = -1
            best_risk = 1e30
            best_mu = -1.0
            best_var = -1.0
            for dev in range(num_devices):
                if dev_ptr[dev] >= num_slots_per_device:
                    continue
                temp_mu = device_mu[dev] + mu[idx] / total_replicas[idx]
                temp_var = compute_updated_device_variance(idx, new_deployment[dev], device_var[dev], var, cov,
                                                           total_replicas)

                risk = temp_mu + z_score * np.sqrt(temp_var)
                if risk < best_risk:
                    best_risk = risk
                    best_dev = dev
                    best_mu = temp_mu
                    best_var = temp_var

            device_mu[best_dev] = best_mu
            device_var[best_dev] = best_var
            new_deployment[best_dev, dev_ptr[best_dev]] = idx
            dev_ptr[best_dev] += 1

    return new_deployment


@njit(fastmath=True, cache=True)
def compute_score(val_data, simulated_replicas, simulated_deployment):
    """
    同时执行:
      - unit_value = val_data / simulated_replicas
      - loads = unit_value[:, simulated_deployment].sum(-1)
    返回:
      loads (T, D)
    """
    T, N = val_data.shape
    D, K = simulated_deployment.shape
    scores = np.empty((T,), dtype=np.float32)
    for t in range(T):
        max_load = 0
        tot_load = 0
        for d in range(D):
            s = 0.0
            for k in range(K):
                idx = simulated_deployment[d, k]
                s += val_data[t, idx] / simulated_replicas[idx]
            tot_load += s
            max_load = max(max_load, s)
        scores[t] = (max_load * D + 1e-2) / (tot_load + 1e-2)

    return np.mean(scores)


def get_score(f, val_data, deployed_replicas,
              current_idx, current_replicas,
              remaind_idx, remaind_replicas):
    # --- 准备输入 ---
    simulated_replicas = deployed_replicas.copy()
    simulated_replicas[current_idx] = current_replicas
    simulated_replicas[remaind_idx] = remaind_replicas
    simulated_deployment = f(simulated_replicas)

    score = compute_score(val_data, simulated_replicas, simulated_deployment)
    return score, simulated_deployment


def neighbor_search(low, high, initial, max_range, get_score, *args):
    """
    从 initial 出发，在 [low, high] 区间内对称搜索最优点（左→右交替）。

    参数:
        low, high   : 搜索边界
        initial     : 初始点
        max_range   : 最大搜索距离
        get_score(x, *args) -> (f, sim): 评分函数，返回目标值与附加信息
        *args       : 传给 get_score 的额外参数

    返回:
        best_x, best_f, best_sim
    """
    # ---- 合法化范围 ----
    max_range = min(max(initial - low, high - initial), max_range)

    # ---- 初始状态 ----
    best_x = initial
    best_score, best_sim = get_score(initial, *args)
    # ---- 对称扩展搜索 ----
    for r in range(1, max_range + 1):
        # 左侧优先
        left = initial - r
        if left >= low:
            score, sim = get_score(left, *args)
            # print(left, score, best_score)
            if score < best_score:
                best_x, best_score, best_sim = left, score, sim

        # 右侧
        right = initial + r
        if right <= high:
            score, sim = get_score(right, *args)
            # print(right, score, best_score)
            if score < best_score:
                best_x, best_score, best_sim = right, score, sim

    return best_x, best_score, best_sim


def flash_tree(X_row, mu, var, cov, num_total_replicas, num_devices, z_score=0.674, deep=3):
    num_experts = mu.shape[0]
    num_avalaible_replicas = num_total_replicas - num_experts

    if deep <= 1:
        default_replicas = np.ones(num_experts, dtype=np.int32)
        default_replicas = min_max_replica(mu, var, num_avalaible_replicas, default_replicas, z_score)[0]
        default_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
        default_deployment = lpt_deployment(
            mu, var, cov, default_deployment, np.zeros(num_experts, dtype=np.int32), default_replicas, z_score
        )
        default_par = compute_score(X_row, default_replicas, default_deployment)
        return default_deployment, default_replicas, default_par

    interval_size = math.ceil(num_experts / deep)
    weight = (mu + z_score * np.sqrt(var))
    idx = np.argsort(-weight)

    deployed_replicas = np.zeros(num_experts, dtype=np.int32)
    deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)

    def _lpt_deployment(replicas):
        nonlocal mu, var, cov, deployment, deployed_replicas, z_score
        return lpt_deployment(mu, var, cov, deployment,
                              np.zeros_like(replicas), replicas, z_score)

    for node in range(deep - 1):
        low, high = 0, num_avalaible_replicas
        simulation_idx = idx[node * interval_size:]
        current_idx = idx[node * interval_size: (node + 1) * interval_size]
        remaind_idx = idx[(node + 1) * interval_size:]

        simulation_replicas = min_max_replica(
            mu[simulation_idx], var[simulation_idx],
            high, np.ones(simulation_idx.shape[0], dtype=np.int32),
            z_score
        )[0]
        current_replicas_f = min_max_replica(
            mu[current_idx], var[current_idx],
            high, np.ones(current_idx.shape[0], dtype=np.int32),
            z_score
        )[1]
        remaind_replicas_f = min_max_replica(
            mu[remaind_idx], var[remaind_idx],
            high, np.ones(remaind_idx.shape[0], dtype=np.int32),
            z_score
        )[1]

        initial_replicas = (simulation_replicas[:interval_size] - 1).sum()

        best_replica, best_score, best_deployment = neighbor_search(
            low, high, initial_replicas, 3,
            lambda mid: get_score(
                _lpt_deployment, X_row, deployed_replicas,
                current_idx, current_replicas_f[mid],
                remaind_idx, remaind_replicas_f[num_avalaible_replicas - mid],
            )
        )

        deployed_replicas[current_idx] = current_replicas_f[best_replica]
        num_avalaible_replicas -= best_replica

        if not num_avalaible_replicas or node == deep - 2:
            deployed_replicas[remaind_idx] = remaind_replicas_f[num_avalaible_replicas]
            break

    final_deployment = -np.ones((num_devices, num_total_replicas // num_devices), dtype=np.int32)
    final_deployment = lpt_deployment(
        mu, var, cov, final_deployment, np.zeros_like(deployed_replicas), deployed_replicas, z_score
    )
    final_par = compute_score(X_row, deployed_replicas, final_deployment)

    return final_deployment, deployed_replicas, final_par


class FlashLB:
    def __init__(
            self,
            max_observation_window=1024,
            update_threshold_ratio=0.95,
            update_threshold_value=0.8,
            update_layers_upper_bound=-1,
            z_upper_quartile=0.75,
    ):
        self.max_observation_window = max_observation_window
        self.update_threshold_ratio = update_threshold_ratio
        self.update_threshold_value = update_threshold_value
        self.update_layers_upper_bound = update_layers_upper_bound
        self.average_to_peak_history = {}
        self.hotness_window = {}
        self.current_deployment = {}
        self.current_deployed_replicas = {}
        self.z_score = stats.norm.ppf(z_upper_quartile)

    @staticmethod
    # @profile
    def compute_statistics(X):
        T, N = X.shape
        mean_ = np.mean(X, axis=0)
        X_centered = X - mean_
        variance_ = np.sum(X_centered ** 2, axis=0) / (T - 1)
        cov_matrix = (X_centered.T @ X_centered) / (T - 1)
        return mean_, variance_, cov_matrix

    @staticmethod
    # @profile
    def sliding_update_stats(mean, cov, x_old, x_new, T):
        assert x_new.shape == x_old.shape
        mean = mean.astype(np.float64, copy=False)
        cov = cov.astype(np.float64, copy=False)
        x_old = x_old.astype(np.float64, copy=False)
        x_new = x_new.astype(np.float64, copy=False)

        sum_old = np.sum(x_old, axis=0)
        sum_new = np.sum(x_new, axis=0)
        deltaS = sum_new - sum_old
        new_mean = mean + deltaS / T

        x_old_centered = x_old - mean
        x_new_centered = x_new - mean

        SA_mu = np.dot(x_old_centered.T, x_old_centered)
        SB_mu = np.dot(x_new_centered.T, x_new_centered)

        Sigma = cov * (T - 1)
        Sigma_new = Sigma + SB_mu - SA_mu - np.outer(deltaS, deltaS) / T
        new_cov = Sigma_new / (T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov

    @staticmethod
    # @profile
    def incremental_update_stats(mean, cov, x_new, T):
        t, N = x_new.shape
        sum_new = np.sum(x_new, axis=0)
        new_T = T + t

        new_mean = (T * mean + sum_new) / new_T

        x_new_centered = x_new - new_mean
        cov_new = cov * (T - 1)
        cov_new += np.dot(x_new_centered.T, x_new_centered)
        cov_new += T * np.outer(mean - new_mean, mean - new_mean)
        new_cov = cov_new / (new_T - 1)

        new_var = np.diag(new_cov)
        return new_mean, new_var, new_cov, new_T

    # @profile
    def register_hotness(self, hotness):
        """
        hotness: np.ndarray, shape [T_total, num_layers, num_experts]
        """
        window_size = self.max_observation_window
        T_total, num_layers, num_experts = hotness.shape

        for layer in range(num_layers):
            # 新样本
            new_X = hotness[-window_size:, layer, :]
            t = new_X.shape[0]

            if layer not in self.hotness_window or t == window_size:
                mu, var, cov = self.compute_statistics(new_X)
                # 环形 buffer 初始化
                buffer = np.zeros((window_size, num_experts), dtype=new_X.dtype)
                buffer[:t] = new_X
                self.hotness_window[layer] = {
                    "buffer": buffer,
                    "start": 0,
                    "length": t,
                    "mean": mu.astype(np.float32, copy=False),
                    "var": var.astype(np.float32, copy=False),
                    "cov": cov.astype(np.float32, copy=False),
                }
                continue

            info = self.hotness_window[layer]
            buf = info["buffer"]
            start = info["start"]
            length = info["length"]
            mu = info["mean"]
            var = info["var"]
            cov = info["cov"]

            # 判断增量或滑动
            if length + t <= window_size:
                mu, var, cov, length = self.incremental_update_stats(
                    mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False), new_X, length)
                end = (start + length - t) % window_size
                buf[end:end + t] = new_X
            else:
                # 滑动窗口
                old_idx = np.arange(start, start + t) % window_size
                x_old = buf[old_idx]
                mu, var, cov = self.sliding_update_stats(
                    mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False), x_old, new_X, window_size)
                # 更新 buffer
                buf[old_idx] = new_X
                start = (start + t) % window_size
                length = window_size

            # 保存更新
            self.hotness_window[layer] = {
                "buffer": buf,
                "start": start,
                "length": length,
                "mean": mu.astype(np.float32, copy=False),
                "var": var.astype(np.float32, copy=False),
                "cov": cov.astype(np.float32, copy=False),
            }

    @staticmethod
    @njit
    def compute_match(src_counts, dst_counts, N, M):
        """
        计算 src_counts 和 dst_counts 之间的匹配矩阵，逐元素计算。
        """
        matches = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(N):
                match = 0
                for k in range(N * M):
                    match += min(src_counts[i, k], dst_counts[j, k])
                matches[i, j] = match
        return matches

    @staticmethod
    def minimize_redeploy_with_inner_permutation(src: np.ndarray, dst: np.ndarray):
        if src.shape != dst.shape:
            raise ValueError("src and dst must have same shape (N, M)")

        N, M = src.shape

        # 预计算每行的 value -> count，使用 NumPy 计数
        src_counts = np.array([np.bincount(row, minlength=M * N) for row in src], dtype=int)
        dst_counts = np.array([np.bincount(row, minlength=M * N) for row in dst], dtype=int)

        # 使用 Numba 加速逐元素匹配计算
        matches = FlashLB.compute_match(src_counts, dst_counts, N, M)

        # 计算代价矩阵：cost[i, j] = M - matches(i, j)
        cost = M - matches

        # 使用匈牙利算法找到最优行匹配
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = list(zip(row_ind.tolist(), col_ind.tolist()))
        total_moves = int(cost[row_ind, col_ind].sum())

        # 根据匹配做行内重排：把 dst 行中能对应到 src 的元素放到 src 相同的 index
        dst_reordered = np.empty_like(dst)
        for src_idx, dst_idx in mapping:
            s_row = src[src_idx]
            d_row = dst[dst_idx]

            # 建立 value -> list of dst positions（用于取出哪一个 dst 上的该值）
            val_to_positions = {}
            for pos, v in enumerate(d_row):
                val_to_positions.setdefault(v, []).append(pos)

            # 结果行初始化及占位标记
            reordered = np.empty(M, dtype=dst.dtype)
            assigned = [False] * M
            used_dst_positions = set()

            # 优先把能对上的元素放回到 src 中对应的位置
            for pos_src, v in enumerate(s_row):
                positions = val_to_positions.get(v)
                if positions:
                    # 取出一个 dst 上该值的位置（pop 一项，保证不重复使用）
                    dst_pos = positions.pop()
                    reordered[pos_src] = v
                    assigned[pos_src] = True
                    used_dst_positions.add(dst_pos)

            # 剩余的 dst 元素（按它们原来的顺序或任意顺序都可以）
            remaining = [d_row[p] for p in range(M) if p not in used_dst_positions]

            # 填补剩余槽位
            ri = 0
            for pos in range(M):
                if not assigned[pos]:
                    reordered[pos] = remaining[ri]
                    ri += 1

            dst_reordered[src_idx] = reordered

        return dst_reordered

    def need_update(self, layer_id=0, val_data=None):
        current_deployment = self.current_deployment.get(layer_id, None)
        if current_deployment is None:
            return True

        if val_data is None:
            val_data = self.hotness_window[layer_id]["buffer"]
        
        average_to_peak_ratio = 1 / compute_score(
            val_data, self.current_deployed_replicas.get(layer_id), current_deployment)
        past_average_to_peak_ratio = self.average_to_peak_history.get(layer_id, 0.0)
        
        return (average_to_peak_ratio < past_average_to_peak_ratio * self.update_threshold_ratio or
                average_to_peak_ratio < self.update_threshold_value)

    def rebalance_experts(self, hotness, num_replicas, num_devices):
        self.register_hotness(hotness)
        _, num_layers, num_expert = hotness.shape

        new_deployment = np.zeros((num_layers, num_devices, num_replicas // num_devices), dtype=np.int32)
        new_deployed_replicas = np.zeros((num_layers, num_expert), dtype=np.int32)
        new_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        delta_average_to_peak_ratio = np.zeros((num_layers,), dtype=np.float32)
        for layer in range(num_layers):
            layer_info = self.hotness_window[layer]
            buf = layer_info["buffer"]
            start = layer_info["start"]
            length = layer_info["length"]
            mu = layer_info["mean"]
            var = layer_info["var"]
            cov = layer_info["cov"]

            idx = np.arange(start, start + length) % self.max_observation_window
            val_data = buf[idx]

            if not self.need_update(layer, val_data):
                new_deployment[layer] = self.current_deployment[layer]
                new_deployed_replicas[layer] = self.current_deployed_replicas[layer]
                new_average_to_peak_ratio[layer] = self.average_to_peak_history.get(layer, 0.0)
                delta_average_to_peak_ratio[layer] = 0
                continue

            deployment, deployed_replicas, new_score = flash_tree(val_data, mu, var, cov, num_replicas, num_devices,
                                                                  z_score=self.z_score)

            new_deployed_replicas[layer] = deployed_replicas
            new_average_to_peak_ratio[layer] = 1 / new_score
            current_deployment = self.current_deployment.get(layer, None)
            if current_deployment is None:
                current_average_to_peak_ratio = 0
                new_deployment[layer] = deployment
            else:
                new_deployment[layer] = FlashLB.minimize_redeploy_with_inner_permutation(current_deployment, deployment)
                current_average_to_peak_ratio = 1 / compute_score(
                    val_data, self.current_deployed_replicas.get(layer), current_deployment)

            # delta average to peak ratio, higher is better
            delta_average_to_peak_ratio[layer] = \
                new_average_to_peak_ratio[layer] - current_average_to_peak_ratio

        priority_idx = np.argsort(-delta_average_to_peak_ratio)
        priority_idx = priority_idx[delta_average_to_peak_ratio[priority_idx] > 0]
        if self.update_layers_upper_bound > 0:
            priority_idx = priority_idx[:self.update_layers_upper_bound]

        for layer in priority_idx:
            self.current_deployment[layer] = new_deployment[layer]
            self.current_deployed_replicas[layer] = new_deployed_replicas[layer]
            self.average_to_peak_history[layer] = new_average_to_peak_ratio[layer]
        return new_deployment, new_deployed_replicas, priority_idx


@njit
def compute_logical_to_physical_map(phy2log, num_layers, num_expert, num_replicas, maxlogcnt):
    log2phy = -1 * np.ones((num_layers, num_expert, maxlogcnt), dtype=np.int64)
    for layer in range(num_layers):
        filled_counts = np.zeros(num_expert, dtype=np.int64)
        for p in range(num_replicas):
            e = phy2log[layer, p]
            rank = filled_counts[e]
            log2phy[layer, e, rank] = p
            filled_counts[e] += 1
    return log2phy


def warmup():
    """
    Run a full warmup to trigger JIT compilation and cache all major kernels.
    Covers:
      - compute_statistics
      - min_max_replica
      - lpt_deployment
      - minimize_redeploy_with_inner_permutation
      - compute_score
      - compute_logical_to_physical_map
    """
    # ---- Basic configuration ----
    num_stages = 128  # number of hotness samples (time dimension)
    num_layers = 1  # single layer warmup
    num_expert = 256  # number of experts per layer
    num_replicas = 320  # total expert replicas
    num_gpus = 64  # total GPUs / devices
    z_score = stats.norm.ppf(0.75)  # 75th percentile for load balancing

    # ---- Generate synthetic hotness data ----
    hotness = np.random.randint(0, 10_000, (num_stages, num_layers, num_expert), dtype=np.int32)
    val_data = hotness[:, 0].astype(np.float32, copy=False)

    # ---- Compute mean / variance / covariance ----
    mean_, var_, cov_ = FlashLB.compute_statistics(val_data)

    # ---- Initialize replica counts (min-max warmup) ----
    total_replicas = np.ones(num_expert, dtype=np.int32)
    total_replicas, *_ = min_max_replica(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        num_replicas - num_expert,
        total_replicas,
        z_score,
    )

    # ---- Initialize deployment matrix (LPT warmup) ----
    experts_per_gpu = num_replicas // num_gpus
    deployment = np.full((num_gpus, experts_per_gpu), -1, dtype=np.int32)

    deployment = lpt_deployment(
        mean_.astype(np.float32, copy=False),
        var_.astype(np.float32, copy=False),
        cov_.astype(np.float32, copy=False),
        deployment,
        np.zeros_like(total_replicas),
        total_replicas,
        z_score,
    )

    # ---- Trigger redeployment matching kernel ----
    FlashLB.minimize_redeploy_with_inner_permutation(deployment, deployment)

    # ---- Trigger score computation kernel ----
    compute_score(val_data, total_replicas, deployment)

    # ---- Trigger logical ↔ physical mapping kernel ----
    phy2log = deployment.reshape((num_layers, -1))
    max_log_cnt = int(total_replicas.max())

    compute_logical_to_physical_map(
        phy2log,
        num_layers,
        num_expert,
        num_replicas,
        max_log_cnt,
    )


warmup()

flash_lb_algo = FlashLB(
    max_observation_window=1024,
    update_threshold_ratio=0.95,
    update_threshold_value=0.8,
    update_layers_upper_bound=16,
    z_upper_quartile=0.75,
)


# @profile
def rebalance_experts(
        weight: torch.Tensor,
        num_replicas: int,
        num_gpus: int,
):
    global flash_lb_algo
    device = weight.device
    hotness = weight.detach().cpu().numpy().astype(np.int32)
    if len(hotness.shape) == 2:
        hotness = np.expand_dims(hotness, axis=0)

    num_iterations, num_layers, num_experts = hotness.shape

    new_deployment, new_deployed_replicas, priority_idx = flash_lb_algo.rebalance_experts(
        hotness, num_replicas, num_gpus
    )

    physical_to_logical_map = new_deployment.reshape((num_layers, -1))
    maxlogcnt = new_deployed_replicas.max()
    logical_to_physical_map = compute_logical_to_physical_map(
        physical_to_logical_map, num_layers, num_experts, num_replicas, maxlogcnt
    )

    # 转为 tensor
    physical_to_logical_map = torch.from_numpy(physical_to_logical_map).to(device)
    log_count = torch.from_numpy(new_deployed_replicas).to(device)
    logical_to_physical_map = torch.from_numpy(logical_to_physical_map).to(device)
    return physical_to_logical_map, logical_to_physical_map, log_count, priority_idx.tolist()


if __name__ == "__main__":
    for it in range(32):
        hotness = torch.randint(1, 1000, (256, 58, 256))
        st = time.time()
        rebalance_experts(hotness, 256 + 64, 64)
        et = time.time()
        print(f"iteration {it}: time {(et - st) * 1000:.2f} ms")
