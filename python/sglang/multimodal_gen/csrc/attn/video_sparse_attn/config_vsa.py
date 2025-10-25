### ADD TO THIS TO REGISTER NEW KERNELS
sources = {"block_sparse": {"source_files": {"h100": "vsa/block_sparse_h100.cu"}}}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ["block_sparse"]

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = "h100"
