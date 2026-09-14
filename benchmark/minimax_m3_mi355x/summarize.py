import json, sys, glob, re, os
result_dir = sys.argv[1]
summary_path = os.path.join(result_dir, "profile_export_aiperf.json")
if not os.path.exists(summary_path):
    print(f"{result_dir}: no summary json"); sys.exit(0)
summary = json.load(open(summary_path))
def metric(tag, stat="avg"):
    entry = summary.get(tag); return None if entry is None else entry.get(stat)
total_tps = metric("total_token_throughput"); output_tps = metric("output_token_throughput"); input_tps = metric("input_token_throughput")
line = (f"{os.path.basename(result_dir)}: dur={metric('benchmark_duration'):.0f}s reqs={metric('request_count'):.0f} "
        f"total_tok/s={total_tps:.0f} ({total_tps/4:.0f}/GPU) out_tok/s={output_tps:.1f} in_tok/s={input_tps:.0f} "
        f"TTFT ms p50={metric('time_to_first_token','p50'):.0f} p90={metric('time_to_first_token','p90'):.0f} p99={metric('time_to_first_token','p99'):.0f} | "
        f"ITL ms p50={metric('inter_token_latency','p50'):.1f} p90={metric('inter_token_latency','p90'):.1f} p99={metric('inter_token_latency','p99'):.1f} | "
        f"E2E s p50={metric('request_latency','p50')/1000:.1f} p90={metric('request_latency','p90')/1000:.1f} p99={metric('request_latency','p99')/1000:.1f} | "
        f"ISL avg={metric('input_sequence_length'):.0f} p50={metric('input_sequence_length','p50'):.0f} p90={metric('input_sequence_length','p90'):.0f} max={metric('input_sequence_length','max'):.0f} | "
        f"OSL avg={metric('output_sequence_length'):.0f} p50={metric('output_sequence_length','p50'):.0f} p90={metric('output_sequence_length','p90'):.0f} | "
        f"per-user out tok/s avg={metric('output_token_throughput_per_user'):.1f}")
# cache accounting from usage metrics if present
for t in ("usage_prompt_tokens", "usage_completion_tokens", "theoretical_prefix_cache_hit"):
    if summary.get(t): line += f" | {t} avg={summary[t].get('avg')}"
# server-side counter deltas over the run
def server_counters(prom_path):
    counters = {}
    for ln in open(prom_path):
        m = re.match(r'^sglang:(prompt_tokens_total|generation_tokens_total|cached_tokens_total|num_requests_total|evicted_tokens_total|num_retracted_requests_total)\{[^}]*\}\s+([0-9.e+]+)', ln)
        if m: counters[m.group(1)] = counters.get(m.group(1), 0) + float(m.group(2))
    return counters
try:
    before = server_counters(os.path.join(result_dir, "server_metrics_before.prom")); after = server_counters(os.path.join(result_dir, "server_metrics_after.prom"))
    prompt = after.get("prompt_tokens_total",0)-before.get("prompt_tokens_total",0); cached = after.get("cached_tokens_total",0)-before.get("cached_tokens_total",0); generated = after.get("generation_tokens_total",0)-before.get("generation_tokens_total",0)
    line += f" | server(incl warmup): prompt={prompt:.0f} cached={cached:.0f} ({(cached/prompt*100) if prompt else 0:.1f}%) fresh_prefill={prompt-cached:.0f} gen={generated:.0f} evicted={after.get('evicted_tokens_total',0)-before.get('evicted_tokens_total',0):.0f} retracted_reqs={after.get('num_retracted_requests_total',0)-before.get('num_retracted_requests_total',0):.0f}"
except Exception as e:
    line += f" | server metrics n/a ({e})"
print(line)
