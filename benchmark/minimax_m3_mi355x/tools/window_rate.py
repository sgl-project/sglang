"""Print the 3600 s window rate (total tok/s per GPU over the profiling records) next to AIPerf's reported value."""
import json, sys
d = sys.argv[1]
recs = [json.loads(l) for l in open(f"{d}/profile_export.jsonl")]
ok = [r for r in recs if r.get("error") is None and r["metadata"]["benchmark_phase"] == "profiling"]
val = lambda r, k: (r["metrics"][k]["value"] if isinstance(r["metrics"][k], dict) else r["metrics"][k])
tokens = sum(val(r, "input_sequence_length") + val(r, "output_sequence_length") for r in ok)
t0 = min(r["metadata"]["request_start_ns"] for r in ok); t1 = max(r["metadata"]["request_end_ns"] for r in ok)
span = (t1 - t0) / 1e9
reported = json.load(open(f"{d}/profile_export_aiperf.json"))["total_token_throughput"]["avg"] / 4
print(f"window rate {tokens / span / 4:,.0f} tok/s/GPU over {span:.0f} s ({len(ok)} requests); aiperf reported {reported:,.0f}")
