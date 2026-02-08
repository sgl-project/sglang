# Installation

## Installing from source

Installing from source requires the latest CUDA Toolkit that matches the major.minor of CUDA Python installed.

Prior to installing the CUTLASS Python interface, one may optionally set the following environment variables:
* `CUTLASS_PATH`: the path to the cloned CUTLASS repository
* `CUDA_INSTALL_PATH`: the path to the installation of CUDA

If these environment variables are not set, the installation process will infer them to be the following:
* `CUTLASS_PATH`: one directory level above the current directory (i.e., `$(pwd)/..`)
* `CUDA_INSTALL_PATH`: the directory holding `/bin/nvcc` for the first version of `nvcc` on `$PATH` (i.e., `which nvcc | awk -F'/bin/nvcc' '{print $1}'`)

**NOTE:** The version of `cuda-python` installed must match the CUDA version in `CUDA_INSTALL_PATH`.

### Installing a developer-mode package
The CUTLASS Python interface can currently be installed via:
```bash
python setup.py develop --user
```
This will allow changes to the Python interface source to be reflected when using the Python interface.

We plan to add support for installing via `python setup.py install` in a future release.

## Docker
To ensure that you have all of the necessary Python modules for running the examples using the
CUTLASS Python interface, we recommend using one of the Docker images located in the docker directory.

For example, to build and launch a container that uses CUDA 12.1 via an NGC PyTorch container, run:
```bash
docker build -t cutlass-cuda12.1:latest -f docker/Dockerfile-cuda12.1-pytorch .
docker run --gpus all -it --rm cutlass-cuda12.1:latest
```

The CUTLASS Python interface has been tested with CUDA 11.8, 12.0, and 12.1 on Python 3.8.10 and 3.9.7.
