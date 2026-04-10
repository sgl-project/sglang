router should contain the original router logic and add one extra logic on choosing between precisions
refer to the reference for implementation guide! the reference's implementation is already tuned to be compatible for both 
    torch.compile
    cudagraph

router should add minimal logical overhead