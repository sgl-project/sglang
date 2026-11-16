| RULER2 subtask | `sgl-kernel` | `flashinfer` | `flashinfer-small-det` |
|---|---:|---:|---:|
| mk_niah_basic | 100.00 | 100.00 | 100.00 |
| mk_niah_easy | 100.00 | 100.00 | 100.00 |
| mk_niah_hard | 90.00 | 90.00 | 90.00 |
| mk_niah_medium | 100.00 | 90.00 | 100.00 |
| mv_niah_basic | 100.00 | 100.00 | 100.00 |
| mv_niah_easy | 81.25 | 88.75 | 92.50 |
| mv_niah_hard | 94.85 | 90.00 | 84.85 |
| mv_niah_medium | 90.00 | 80.00 | 94.59 |
| qa_basic | 100.00 | 100.00 | 100.00 |
| qa_easy | 95.00 | 95.00 | 95.00 |
| qa_hard | 75.42 | 73.75 | 73.75 |
| qa_medium | 85.00 | 80.00 | 85.00 |

Sampling for every run and eval: `{"chat_template_kwargs": {"thinking": true}, "extra_body": null, "max_tokens": null, "min_p": 0.0, "reasoning_effort": "high", "repetition_penalty": 1.0, "seed": null, "system_message": null, "temperature": 1.0, "top_p": 0.95}`

`sgl-kernel` (`r5-sgl-kernel`):

```text
name=r5-sgl-kernel gpus=0,1,2,3 port=30000
env=
server_args=
tree_core=python rust_build_mode=never
2026-11-15T13:00:48Z
healthy 2026-11-15T13:06:30Z
gsm8k exit=0 2026-11-15T13:12:22Z
ruler2 exit=0 2026-11-15T16:06:55Z
gsm8k: sgl_eval_version=0.1.3 num_examples=1319 num_threads=64 latency_seconds=350.1
{"error_rate": 0.0, "score": 0.9666413949962093, "stop_rate": 1.0, "truncated_rate": 0.0}
ruler2: sgl_eval_version=0.1.3 num_examples=240 num_threads=4 latency_seconds=10471.3
{"error_rate": 0.0, "score": 0.9262663398692813, "stop_rate": 0.9916666666666667, "task.mk_niah_basic": 1.0, "task.mk_niah_easy": 1.0, "task.mk_niah_hard": 0.9, "task.mk_niah_medium": 1.0, "task.mv_niah_basic": 1.0, "task.mv_niah_easy": 0.8125, "task.mv_niah_hard": 0.9485294117647058, "task.mv_niah_medium": 0.9, "task.qa_basic": 1.0, "task.qa_easy": 0.95, "task.qa_hard": 0.7541666666666668, "task.qa_medium": 0.85, "truncated_rate": 0.008333333333333333}
```

`flashinfer` (`r5-flashinfer`):

```text
name=r5-flashinfer gpus=4,5,6,7 port=30001
env=
server_args=--dsa-topk-backend flashinfer
tree_core=python rust_build_mode=never
2026-11-15T15:54:21Z
healthy 2026-11-15T16:01:13Z
gsm8k exit=0 2026-11-15T16:05:59Z
ruler2 exit=0 2026-11-15T18:38:37Z
gsm8k: sgl_eval_version=0.1.3 num_examples=1319 num_threads=64 latency_seconds=284.5
{"error_rate": 0.0, "score": 0.9696739954510993, "stop_rate": 1.0, "truncated_rate": 0.0}
ruler2: sgl_eval_version=0.1.3 num_examples=240 num_threads=4 latency_seconds=9156.9
{"error_rate": 0.0, "score": 0.90625, "stop_rate": 0.9875, "task.mk_niah_basic": 1.0, "task.mk_niah_easy": 1.0, "task.mk_niah_hard": 0.9, "task.mk_niah_medium": 0.9, "task.mv_niah_basic": 1.0, "task.mv_niah_easy": 0.8875, "task.mv_niah_hard": 0.9, "task.mv_niah_medium": 0.8, "task.qa_basic": 1.0, "task.qa_easy": 0.95, "task.qa_hard": 0.7375, "task.qa_medium": 0.8, "truncated_rate": 0.0125}
```

`flashinfer-small-det` (`r5-flashinfer-small-det`):

```text
name=r5-flashinfer-small-det gpus=0,1,2,3 port=30000
env=SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1
server_args=--dsa-topk-backend flashinfer
tree_core=python rust_build_mode=never
2026-11-15T16:07:01Z
healthy 2026-11-15T16:12:22Z
gsm8k exit=0 2026-11-15T16:17:50Z
ruler2 exit=0 2026-11-15T16:52:14Z
gsm8k: sgl_eval_version=0.1.3 num_examples=1319 num_threads=64 latency_seconds=326.3
{"error_rate": 0.0, "score": 0.9711902956785443, "stop_rate": 1.0, "truncated_rate": 0.0}
ruler2: sgl_eval_version=0.1.3 num_examples=240 num_threads=4 latency_seconds=2063.3
{"error_rate": 0.0, "score": 0.9297479464758878, "stop_rate": 1.0, "task.mk_niah_basic": 1.0, "task.mk_niah_easy": 1.0, "task.mk_niah_hard": 0.9, "task.mk_niah_medium": 1.0, "task.mv_niah_basic": 1.0, "task.mv_niah_easy": 0.925, "task.mv_niah_hard": 0.8485294117647058, "task.mv_niah_medium": 0.9459459459459459, "task.qa_basic": 1.0, "task.qa_easy": 0.95, "task.qa_hard": 0.7375, "task.qa_medium": 0.85, "truncated_rate": 0.0}
```
