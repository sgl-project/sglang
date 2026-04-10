we need to profile on both kernels with varying batch sizes
assume to use qwen3 30b a3b, use the expert's linear weight size and group size

we need two kinds of profiling:
    1. just test the kernel with uniform token distribution across experts 
    2. use routing information (less priority)

the ultimate goal of this profile is to show the (batch size) knee of memory bound and compute bound of different configs {a8w8, a16w4, a16w16}
you should create necessary plots for demostration
