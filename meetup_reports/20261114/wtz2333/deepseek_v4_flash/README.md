# DeepSeek-V4-Flash NPU MeetUp report

Participant: `wtz2333`. Report ID: `DSV4F-NPU-20261114-01`. Test date: 2026-11-14.

- [Deployment and tuning report](DeepSeek-V4-Flash_MeetUp_Report_20261114.md)
- [Complete report and reproduction attachments](DeepSeek-V4-Flash_MeetUp_Report_20261114.zip)

The attachment archive contains 15 files, including full launch/benchmark/accuracy scripts, console transcripts, a transcribed summary, English feedback drafts, and a checksum manifest.

Measured results: Mean TPOT 9.45 ms, P99 TPOT 26.97 ms, output throughput 644.41 tokens/s, and observed prefix-cache hit rate 89.1%. Mean TPOT meets both 20 ms and 50 ms budgets; P99 meets 50 ms and does not meet 20 ms. GSM8K accuracy is 10/10 on the reported sample, with no truncation or request errors. The two budgets reuse one configuration and one measurement run.

The report distinguishes nominal input length from observed token counts. Original server JSONL/metrics/prediction files and detailed environment provenance remain to be attached. English feedback files are drafts, not submitted issues.
