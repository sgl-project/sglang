## What to do if server crash?

If you see "Illegal memory access", try a smaller value for `--mem-frac`.
If you see server hangs, try `--disable-flashinfer-sampling`. If it still does not work, try `--disable-cuda-graph`.
