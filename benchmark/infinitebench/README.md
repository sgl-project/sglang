## Run benchmark

### SGLang
#### Set up environment
```bash
conda create -n sglang python=3.10
conda activate sglang
pip install --upgrade pip
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
```

#### Launch server and run eval
```
python -m sglang.launch_server --model-path gradientai/Llama-3-8B-Instruct-Gradient-1048k --port 30000 --max-total-tokens 131072
```

In a separate terminal, run eval on the first 10 samples as follows
```
python bench_sglang.py --task passkey --num-samples 10
```

### TensorRT
The following evaluation with TensorRT has been tested with 1xH100 (80 GB SXM5).

#### Set up enviroment
```bash
conda create -n tensorrt python=3.10
conda activate tensorrt
conda install -c conda-forge mpi4py openmpi
sudo apt-get -y install libopenmpi-dev && pip install tensorrt_llm==0.15.0
```

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install --upgrade -r requirements-dev.txt
cd examples/llama
```

#### Download dataset
The following commands will download the dataset files in `./data` directory, which is hardcoded in the script.
```bash
URL="https://raw.githubusercontent.com/OpenBMB/InfiniteBench/51d9b37b0f1790ead936df2243abbf7f0420e439/scripts/download_dataset.sh"
wget $URL -O download_infinitebench.sh
bash download_infinitebench.sh
```

#### Prepare checkpoint
```bash
sudo apt install git-lfs
git-lfs clone https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k/

python convert_checkpoint.py \
    --model_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
    --output_dir /tmp/llama-3-8B-1048k/trt_ckpts \
    --dtype float16

```

#### Build engine and run eval
```bash
python -m tensorrt_llm.commands.build \
    --checkpoint_dir /tmp/llama-3-8B-1048k/trt_ckpts \
    --output_dir /tmp/llama-3-8B-1048k/trt_engines \
    --gemm_plugin float16 \
    --max_num_tokens 4096 \
    --max_input_len 131072 \
    --max_seq_len 131082 \
    --use_paged_context_fmha enable

python ../eval_long_context.py \
    --task passkey \
    --engine_dir /tmp/llama-3-8B-1048k/trt_engines \
    --tokenizer_dir ./Llama-3-8B-Instruct-Gradient-1048k/ \
    --stop_idx 10 \
    --max_input_length 131072 \
    --enable_chunked_context \
    --max_tokens_in_paged_kv_cache 131136 \
    --data_dir ./data \
    --output_dir ./
```
