"""Native fused manager validation; use --help for the current fixture matrix."""

from bench_lim_framework import main

if __name__ == "__main__":
    main(default_widths="2,3,4,5,8,14", default_timing=False)
