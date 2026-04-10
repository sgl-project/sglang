priority of tasks:

0. read the .hetermoe and understand all .md

    (collect and present results)
    (wait until further instructions)

1. implementing the system (migrate from trtllm's mix {bf16, nvfp4} to sglang's mix{marlin int4, int8})
2. evaluating the implementation kernel-wise with varying fake inputs&weights
    a. usability the implementation should run
    b. efficiency: the mixture of precision should show performance speedup compared to naive bf16
        and the speedup should be reasonable (lies between the two extreme precisions)
    c. accuracy: the implementation should provide negligible accuracy loss (MSE)
        show that the MSE of two gemm kernel lies between two extreme precisions
    d. make necessary plots and tables to demonstrate efficiency and accuracy
3. collect the routing stats for real load imbalance
4. evaluating the implementation layer-wise with collected real routing stats but still fake inputs&weights
    plese make plots and tables to demonstrate efficiency and accuracy

    (collect and present results)
    (wait until further instructions)

5. download/quantize the weights for lower precisions
6. evaluating the implementation E2E:
    a. accuracy metrics of both perplexity and downstream tasks performance (prioritize shorter benchmarks like hellaswag and gsm8k)
    b. efficiency: TTFT, ITL (inter token latency)


versioning and logging:
after each feature and test is implemented, commit and push with informative messages
after the test passed and feature is corrected, commit and push

make sure you log in some path that include the information I should be aware of
    these logs and commits should be pointed to each other when needed