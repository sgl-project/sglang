# rdma_tail_0706 raw data

Tail RDMA single-NIC small-flow experiment on cloud-099 -> cloud-100.

- `source/`: initiator-side raw samples, summaries, aggregate CSVs, driver log.
- `target/`: target-side logs/json and RDMA receive monitor CSVs.
- `tidy/`: derived comparison tables used by the report.
- Sizes: `1MiB..32MiB`, step `1MiB`; runs: `tail_rdma_small_1x100`, `tail_rdma_small_1x200`.
