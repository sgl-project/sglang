because there will be multiple weight precisions:
w16, w8, w4

we need to support the loading of model weights together into VRAM
    the groupgemm of compute may need to compute on different expert weights

follow reference in reference.md for more details on their weight loading

different from the reference code where no TP or EP is supported, here we need to support both TP and EP:
    in ep the default assignment is to have the different precisions of the experts on the same rank
        however, we need to support different assignments, for example, users can custome the precision assignment like:
            INT4: {rank 0: experts 0-15, ...}
            this leaves potential redundancy of low bit precision weights to speed up by mitigating load imbalance
    this should have low priority since our main focus is to use different precisions for different experts
    *** for now we can first assume everything is done on a single rank...

similar to reference, you should have one flag indicating we want to use heter moe instead of the vanilla, we should have flags on 
    1. what precision schemes are of our interest, for example {"a8w8", "a16w4"}
    2. what kind of criteria: default is token count, and we should be satisfied with that primarily
    3. percentage of each precision
(all similar to reference)
