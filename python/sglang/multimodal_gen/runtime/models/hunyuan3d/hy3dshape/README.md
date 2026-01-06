# Hunyuan3D-2.1-Shape

## Quick Inference

Given a reference image `image.png`, you can run inference using the following code. The result will be saved as `demo.glb`.

```bash
python3 minimal_demo.py
```

**Memory Recommendation:** For we recommend using a GPU with at least **10GB VRAM**.

# Training

Here we demonstrate the complete training workflow of DiT on a small dataset.

## Data Preprocessing

The rendering and watertight mesh generation process is described in detail in [this document](tools/README.md). After preprocessing, the dataset directory structure should look like the following:

```yaml
dataset/preprocessed/{uid}
├── geo_data
│   ├── {uid}_sdf.npz
│   ├── {uid}_surface.npz
│   └── {uid}_watertight.obj
└── render_cond
    ├── 000.png
    ├── ...
    ├── 023.png
    ├── mesh.ply
    └── transforms.json
```

We provide a preprocessed mini_dataset containing 8 cases (all sourced from Objaverse-XL) as `tools/mini_trainset`, which can be used directly for DiT overfitting training experiments.

## Launching Training

We provide example configuration files and launch scripts for reference. By default, the training runs on a single node with 8 GPUs using DeepSpeed. Users can modify the configurations and scripts as needed to suit their environment.

Configuration File
```
configs/hunyuandit-mini-overfitting-flowmatching-dinog518-bf16-lr1e4-512.yaml
```
Launch Script

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip

# export config=configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
# export output_dir=output_folder/dit/fintuning_lr1e5
export config=configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml
export output_dir=output_folder/dit/overfitting_depth_16_token_4096_lr1e4

bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
```
