import requests
import statistics

URL = "http://127.0.0.1:30100/server_info"

info = requests.get(URL, timeout=60).json()
states = info.get("internal_states", [])

global_actual = 0
global_static = 0
global_graph = 0
global_local_tier = 0
global_dp_tier = 0
global_steps = 0

print("=" * 110)
print("DSpark Compact Scheduling Analysis")
print("=" * 110)

for rank, state in enumerate(states):
    payload = state.get("dspark_info_record")

    if not payload:
        print(f"DP{rank}: no dspark_info_record")
        continue

    gamma = int(payload.get("verify_num_draft_tokens", 6))

    records = [
        r for r in payload.get("records", [])
        if r.get("mode") == "compact"
        and r.get("num_running_reqs", 0) > 0
        and r.get("reqs")
    ]

    if not records:
        print(f"DP{rank}: no valid compact decode records")
        continue

    actual_list = []
    static_list = []
    graph_list = []
    local_tier_list = []
    dp_tier_list = []

    verify_len_hist = {}

    for r in records:
        reqs = r.get("reqs") or []

        # 真正 compact scheduler 给每个 request 分配的 verify_len
        actual = sum(int(req["verify_len"]) for req in reqs)

        # 如果不用 compact，每个 request 固定 verify gamma 个
        static = len(reqs) * gamma

        # 实际 replay 的 graph token tier
        graph = int(r.get("verify_tokens_graph_key", actual))

        local_tier = int(r.get("verify_tokens_local", actual))

        dp_tier = int(r.get("verify_tokens_dp_synced", -1))

        actual_list.append(actual)
        static_list.append(static)
        graph_list.append(graph)
        local_tier_list.append(local_tier)

        if dp_tier >= 0:
            dp_tier_list.append(dp_tier)

        for req in reqs:
            v = int(req["verify_len"])
            verify_len_hist[v] = verify_len_hist.get(v, 0) + 1

    actual_sum = sum(actual_list)
    static_sum = sum(static_list)
    graph_sum = sum(graph_list)

    global_actual += actual_sum
    global_static += static_sum
    global_graph += graph_sum
    global_local_tier += sum(local_tier_list)
    global_dp_tier += sum(dp_tier_list)
    global_steps += len(records)

    reduction = 1.0 - actual_sum / static_sum

    print(
        f"DP{rank:02d}: "
        f"steps={len(records):4d}, "
        f"avg actual={statistics.mean(actual_list):6.2f}, "
        f"avg static={statistics.mean(static_list):6.2f}, "
        f"avg graph={statistics.mean(graph_list):6.2f}, "
        f"logical reduction={reduction * 100:6.2f}%"
    )

    print(
        f"       verify_len distribution: "
        f"{dict(sorted(verify_len_hist.items()))}"
    )

print("=" * 110)

if global_static > 0:
    reduction = 1.0 - global_actual / global_static

    graph_padding_vs_actual = (
        global_graph / global_actual - 1.0
        if global_actual > 0 else 0
    )

    graph_vs_static = (
        global_graph / global_static - 1.0
        if global_static > 0 else 0
    )

    print(f"Total decode steps             : {global_steps}")
    print(f"Static logical verify tokens   : {global_static}")
    print(f"Compact scheduled tokens       : {global_actual}")
    print(f"Graph replay token slots       : {global_graph}")

    print()
    print(f"Compact logical reduction      : {reduction * 100:.2f}%")
    print(f"Graph padding vs compact       : {graph_padding_vs_actual * 100:.2f}%")
    print(f"Graph slots vs raw static      : {graph_vs_static * 100:.2f}%")