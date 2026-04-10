we only require naive profiling at this stage:
    for different batch size, for each layer, what is the load per expert 

for datasets, we can simply use share gpt
for snapshots, take one snapshot for prefill and one snapshot for decoding
we need the load imbalance information for all layers in the model

but we don't need too much data points, simply pick batch size from [2^0, ..., 2^10]
 and one prefill one decode for each datapoint

each datapoint is a file, in such format 
{
    "transformer_block_{i}": [] : list of 128 interger numbers
}

all the datapoints should be collected in a folder with filenames indicating when(batch size; prefill/decode) is the data collected
