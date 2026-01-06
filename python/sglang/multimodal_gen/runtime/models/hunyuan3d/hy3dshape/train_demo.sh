export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8
# export CUDA_VISIBLE_DEVICES=0
# export num_gpu_per_node=1

export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip

# export config=configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
# export output_dir=output_folder/dit/fintuning_lr1e5
export config=configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml
export output_dir=output_folder/dit/overfitting_depth_16_token_4096_lr1e4

bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
