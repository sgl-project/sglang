# HF3FS benchmarks

These scripts exercise the HF3FS client and SGLang's HiCache storage adapter
against an existing 3FS mount. They use synthetic tensors; no model checkpoint,
inference server, or dataset is needed.

| Script | What it measures | Default `--file-path` |
| --- | --- | --- |
| [bench_client.py](bench_client.py) | Raw client reads and writes at byte offsets | `/data/bench.bin` |
| [bench_storage.py](bench_storage.py) | Cache-key lookup/allocation and tensor reads/writes through the adapter | `/data/test.bin` |
| [bench_zerocopy.py](bench_zerocopy.py) | Public adapter APIs reading/writing a registered, page-first host KV pool | `/data/bench_zerocopy.bin` |

## Prerequisites

- Install SGLang and its serving dependencies from this checkout; see the
  [installation instructions](../../README.md).
- Set up a running 3FS filesystem and its Python USRBIO client (`hf3fs_fuse.io`);
  see [HF3FS setup](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md)
  and [client installation](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/setup_usrbio_client.md).
- Use an existing, writable directory on the 3FS mount. The destination file
  does not need to exist. An ordinary local filesystem is rejected by the real
  client; these scripts do not select a mock client.
- Provide a C++ compiler and Ninja. Importing the client builds the
  `hf3fs_utils` tensor-copy extension with PyTorch.
- Provide enough RAM and `/dev/shm` capacity for the buffers below. In a
  container, the application and 3FS FUSE client must be able to access the same
  shared-memory files.
- `bench_zerocopy.py` also requires a CUDA GPU and `torch_memory_saver`. Its
  GPU pool is used to construct the host pool; the timed transfers are between
  CPU memory and 3FS, not between GPU and CPU memory.

The adapter benchmarks use in-process `Hf3fsLocalMetadataClient` instances.
They do not require the separate SGLang HTTP metadata server or
`SGLANG_HICACHE_HF3FS_CONFIG_PATH`.

## Run

Run from the repository root with the appropriate Python environment active.
Use dedicated benchmark filenames: each script creates or resizes the supplied
file, overwrites data, and deletes it after a successful run. A failed run can
leave the file behind. Do not point these scripts at an existing serving cache
or other data you need to retain.

Set `hf3fs_bench_dir` to your existing writable directory on 3FS:

```bash
hf3fs_bench_dir=/path/to/writable/3fs-directory
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# For installations with the native API library bundled in the Python wheel.
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 benchmark/hf3fs/bench_client.py \
  --file-path "$hf3fs_bench_dir/client.bin"

python3 benchmark/hf3fs/bench_storage.py \
  --file-path "$hf3fs_bench_dir/storage.bin"
```

Other native library dependencies must also be discoverable by the loader;
follow the client installation instructions for your environment.

The zero-copy script enables the memory saver. Configure its preload library
in the child process before running the script:

```bash
python3 - --file-path "$hf3fs_bench_dir/zerocopy.bin" <<'PY'
import subprocess
import sys

from torch_memory_saver import configure_subprocess

with configure_subprocess():
    subprocess.run(
        [sys.executable, "benchmark/hf3fs/bench_zerocopy.py", *sys.argv[1:]],
        check=True,
    )
PY
```

### Workload options

All scripts accept `--file-path` and the following options. Defaults preserve
the original workloads. Use `--help` for each script's options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--file-size` | `1099511627776` (1 TiB) | Logical file size in bytes for the throughput phase |
| `--numjobs` | `16` | Number of clients |
| `--entries` | `8` | I/O ring entries per client in the throughput phase |

Client and storage benchmarks also accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bytes-per-page` | `16777216` (16 MiB) | Bytes per client I/O block or storage cache slot |
| `--warmup` | `50` | Warmup iterations per direction |
| `--iterations` | `100` | Measured iterations per direction |
| `--batch-pages` | `128` | Storage benchmark only: cache slots transferred per iteration |

The client transfers `numjobs * entries` blocks per iteration. Its correctness
test checks up to 128 blocks, using 32 ring entries. The storage correctness
test always uses eight cache slots and two ring entries per client; its file
size is derived from `--bytes-per-page` independently of `--file-size`.

The zero-copy benchmark accepts these options instead of byte page sizes and
iteration counts:

| Option | Default | Meaning |
| --- | --- | --- |
| `--page-size` | `64` | Tokens per logical KV page |
| `--tokens-per-batch` | `1024` | Tokens transferred per batch |
| `--num-batches` | `16` | Batches per direction |

Its model geometry, dtype, and host-to-device capacity ratio stay fixed.
GPU pool capacity is `tokens_per_batch * num_batches`; page bytes are derived
from the pool geometry. Each token occupies 256 KiB of K/V data.

Sizes and counts must be positive; warmup may be zero. Byte page sizes must
be divisible by the bfloat16 element size (two bytes). The client file must
hold `(warmup + iterations) * numjobs * entries` blocks. The storage file must
hold at least `batch_pages` slots; older keys may be evicted as writes fill it.
Zero-copy batches must be divisible by the token page size, and the file must
hold all K/V data written in the run.

## Workloads and results

### Raw client

With default options, `bench_client.py` first writes and reads 128 distinct
16 MiB blocks and checks exact tensor equality. It then uses 16 clients with
eight I/O ring entries each, with independent read and write tensors per client.

Each throughput iteration transfers 2 GiB at randomized block-aligned offsets.
There are 50 warmup and 100 measured iterations per direction: 300 GiB written
and 300 GiB read, plus 2 GiB per direction for the initial correctness test.
The read phase uses the same offsets as the write phase.

### Storage adapter

With default options, `bench_storage.py` first uses a 128 MiB file with eight
16 MiB cache slots.
It checks single-key writes, eviction of the oldest keys when full, reads,
deletion and slot reuse, clearing metadata, and a batch round trip. Tensor
contents are compared with `torch.equal`.

The throughput phase resizes the same file to 1 TiB. It writes 128 distinct
16 MiB tensors under new keys per iteration, using 16 clients and eight ring
entries per client. Reads randomly sample existing keys into reusable buffers.
It performs 50 warmup and 100 measured iterations per direction, transferring
300 GiB in each direction, in addition to the correctness test.

The script calls the adapter's internal `_batch_set()` and `_batch_get()`
helpers with explicit tensors. Timings include metadata processing and I/O.
The throughput phase checks operation success, while `test()` checks contents.

### Host-pool / zero-copy path

`bench_zerocopy.py` uses `batch_set_v1()` and `batch_get_v1()` after registering
an `MHATokenToKVPoolHost` with the adapter. It does not create a cache controller.
Its synthetic MHA geometry has 64 layers, eight KV heads, head dimension 128,
and `bfloat16` data.

By default, `page_size = 64` means 64 tokens per logical KV page. Each logical
page contains 16 MiB of K/V data; this HF3FS path stores separate 8 MiB K and V
components. There are 16 operations of 1,024 tokens each: 256 logical pages,
or 4 GiB per direction.

Host pages are initialized before timing. The script performs one write pass
and one read pass, checks their success flags, and prints elapsed seconds and
aggregate GiB/s. It has no warmup loop or tensor-content comparison.

The `page_first` layout lets the adapter use views of contiguous K and V page
regions, avoiding temporary packing/unpacking buffers. The USRBIO client still
copies between these views and shared memory; the complete I/O path is not
copy-free.

## Capacity and interpretation

By default, all throughput phases use a 1 TiB **logical** file. Resizing
establishes its length without writing 1 TiB of payload. The default client and
adapter runs subsequently write approximately 300 GiB each, so account for real storage
capacity as well as the logical file size.

Approximate default benchmark buffer sizes, excluding padding, library
overhead, and additional GPU staging buffers:

| Script | CPU tensors / host KV pool | Shared-memory buffers | GPU KV buffers |
| --- | --- | --- | --- |
| `bench_client.py` | 4 GiB | 4 GiB | None |
| `bench_storage.py` | 4 GiB | 4 GiB | None |
| `bench_zerocopy.py` | 8 GiB | 2 GiB | 4 GiB |

The host-pool constructor also leaves a reserve of available host RAM, and can
allocate GPU transfer staging buffers. The table is not a complete minimum
machine-memory requirement.

Client and storage-adapter statistics report mean, min, p25, p50, p75, and max
for the measured throughput samples (100 by default), in **GiB/s**
(`2^30` bytes per GiB).
Timings include client processing and memory copies, rather than isolating
disk or network bandwidth. They wait for I/O completion without an explicit
`fsync()` in the timed region.

## Functional checks and troubleshooting

For small functional runs, use the command-line options:

```bash
python3 benchmark/hf3fs/bench_client.py \
  --file-path "$hf3fs_bench_dir/client-small.bin" \
  --file-size 8388608 --bytes-per-page 65536 \
  --numjobs 2 --entries 2 --warmup 1 --iterations 3

python3 benchmark/hf3fs/bench_storage.py \
  --file-path "$hf3fs_bench_dir/storage-small.bin" \
  --file-size 65536 --bytes-per-page 4096 --batch-pages 4 \
  --numjobs 2 --entries 2 --warmup 1 --iterations 3
```

For zero-copy, use the same preload setup:

```bash
python3 - --file-path "$hf3fs_bench_dir/zerocopy-small.bin" \
  --file-size 16777216 --numjobs 2 --entries 2 \
  --page-size 4 --tokens-per-batch 8 --num-batches 2 <<'PY'
import subprocess
import sys

from torch_memory_saver import configure_subprocess

with configure_subprocess():
    subprocess.run(
        [sys.executable, "benchmark/hf3fs/bench_zerocopy.py", *sys.argv[1:]],
        check=True,
    )
PY
```

This zero-copy workload transfers 4 MiB per direction while retaining the
same model geometry and CUDA/host-pool path. It still requires the host RAM
reserve and additional GPU staging allocations described above.

All three scripts passed reduced workloads configured through the CLI against
a real local 3FS mount using Python 3.12, PyTorch 2.14.1 with CUDA 13.0, and
an RTX 5050. An earlier additional check cleared the zero-copy host pages after writing and verified
exact K/V restoration after reading. This establishes functional coverage;
default-size workloads and representative performance were not measured.

- **`TorchMemorySaver observes invalid LD_PRELOAD`:** use the zero-copy launch
  wrapper above.
- **Duplicate logging flags while loading the client:** avoid mixing the wheel's
  native API library with dependencies from a different 3FS build. Check
  `LD_LIBRARY_PATH` ordering and use matching native libraries.
- **`close HiCacheHF3FS: [Errno 9] Bad file descriptor` at process exit:** the
  current adapter's `atexit` callback can close clients that the benchmark
  already closed. This was observed after successful local runs; inspect prior
  I/O errors and the process exit status separately.
