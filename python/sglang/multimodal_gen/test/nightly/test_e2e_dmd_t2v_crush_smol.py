import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from sgl_diffusion.test.ssim.test_inference_similarity import (
    compute_video_ssim_torchvision,
)

# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

NUM_NODES = "1"
MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# preprocessing
DATA_DIR = "data"
LOCAL_RAW_DATA_DIR = Path(os.path.join(DATA_DIR, "crush-smol"))
NUM_GPUS_PER_NODE_PREPROCESSING = "1"
PREPROCESSING_ENTRY_FILE_PATH = "sgl_diffusion/pipelines/preprocess/v1_preprocess.py"

LOCAL_PREPROCESSED_DATA_DIR = Path(os.path.join(DATA_DIR, "crush-smol_processed_t2v"))


# training
NUM_GPUS_PER_NODE_TRAINING = "4"
TRAINING_ENTRY_FILE_PATH = "sgl_diffusion/training/wan_distillation_pipeline.py"
LOCAL_TRAINING_DATA_DIR = os.path.join(
    LOCAL_PREPROCESSED_DATA_DIR, "combined_parquet_dataset"
)
LOCAL_VALIDATION_DATASET_FILE = (
    "examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json"
)
LOCAL_OUTPUT_DIR = Path(os.path.join(DATA_DIR, "outputs"))


def download_data():
    # create the data dir if it doesn't exist
    data_dir = Path(DATA_DIR)

    print(f"Creating data directory at {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading raw dataset to {LOCAL_RAW_DATA_DIR}...")
    try:
        result = snapshot_download(
            repo_id="wlsaidhi/crush-smol-merged",
            local_dir=str(LOCAL_RAW_DATA_DIR),
            repo_type="dataset",
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),  # In case authentication is needed
        )
        print(f"Download completed successfully. Files downloaded to: {result}")

        # Verify the download
        if not LOCAL_RAW_DATA_DIR.exists():
            raise RuntimeError(
                f"Download appeared to succeed but {LOCAL_RAW_DATA_DIR} does not exist"
            )

        # List downloaded files
        print("Downloaded files:")
        for file in LOCAL_RAW_DATA_DIR.rglob("*"):
            if file.is_file():
                print(f"  - {file.relative_to(LOCAL_RAW_DATA_DIR)}")

    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise


def run_preprocessing():
    # remove the local_preprocessed_data_dir if it exists
    if LOCAL_PREPROCESSED_DATA_DIR.exists():
        print(f"Removing local_preprocessed_data_dir: {LOCAL_PREPROCESSED_DATA_DIR}")
        shutil.rmtree(LOCAL_PREPROCESSED_DATA_DIR)

    # Run torchrun command
    cmd = [
        "torchrun",
        "--nnodes",
        NUM_NODES,
        "--nproc_per_node",
        NUM_GPUS_PER_NODE_PREPROCESSING,
        PREPROCESSING_ENTRY_FILE_PATH,
        "--model_path",
        MODEL_PATH,
        "--seed",
        "42",
        "--data_merge_path",
        os.path.join(LOCAL_RAW_DATA_DIR, "merge.txt"),
        "--preprocess_video_batch_size",
        "1",
        "--max_height",
        "480",
        "--max_width",
        "832",
        "--num_frames",
        "81",
        "--dataloader_num_workers",
        "0",
        "--output_dir",
        LOCAL_PREPROCESSED_DATA_DIR,
        "--train_fps",
        "16",
        "--samples_per_file",
        "1",
        "--flush_frequency",
        "1",
        "--video_length_tolerance_range",
        "5",
        "--preprocess_task",
        "t2v",
    ]

    process = subprocess.run(cmd, check=True)


def run_training():
    cmd = [
        "torchrun",
        "--nnodes",
        NUM_NODES,
        "--nproc_per_node",
        NUM_GPUS_PER_NODE_TRAINING,
        TRAINING_ENTRY_FILE_PATH,
        "--model_path",
        MODEL_PATH,
        "--inference_mode",
        "False",
        "--pretrained_model_name_or_path",
        MODEL_PATH,
        "--data_path",
        LOCAL_TRAINING_DATA_DIR,
        "--validation_dataset_file",
        LOCAL_VALIDATION_DATASET_FILE,
        "--train_batch_size",
        "1",
        "--num_latent_t",
        "8",
        "--num_gpus",
        NUM_GPUS_PER_NODE_TRAINING,
        "--sp_size",
        "1",
        "--tp_size",
        "1",
        "--hsdp_replicate_dim",
        "1",
        "--hsdp_shard_dim",
        NUM_GPUS_PER_NODE_TRAINING,
        "--train_sp_batch_size",
        "1",
        "--dataloader_num_workers",
        "10",
        "--gradient_accumulation_steps",
        "1",
        "--max_train_steps",
        "501",
        "--learning_rate",
        "2e-6",
        "--fake_score_learning_rate",
        "2e-6",
        "--mixed_precision",
        "bf16",
        "--training_state_checkpointing_steps",
        "1000",
        "--weight_only_checkpointing_steps",
        "1000",
        "--validation_steps",
        "50",
        "--validation_sampling_steps",
        "3",
        "--log_validation",
        "--checkpoints_total_limit",
        "3",
        "--ema_start_step",
        "0",
        "--training_cfg_rate",
        "0.0",
        "--output_dir",
        LOCAL_OUTPUT_DIR,
        "--tracker_project_name",
        "ci_wan_t2v_dmd_overfit",
        "--num_height",
        "480",
        "--num_width",
        "832",
        "--num_frames",
        "81",
        "--flow_shift",
        "8",
        "--validation_guidance_scale",
        "6.0",
        "--weight_decay",
        "0.01",
        "--generator_update_interval",
        "5",
        "--dmd_denoising_steps",
        "1000,757,522",
        "--min_timestep_ratio",
        "0.02",
        "--max_timestep_ratio",
        "0.98",
        "--seed",
        "1000",
        "--real_score_guidance_scale",
        "3.5",
        "--dit_precision",
        "fp32",
        "--max_grad_norm",
        "1.0",
        "--enable_gradient_checkpointing_type",
        "full",
    ]

    print(f"Running training with command: {cmd}")
    process = subprocess.run(cmd, check=True)


def test_e2e_overfit_single_sample():
    os.environ["WANDB_MODE"] = "online"

    download_data()
    run_preprocessing()
    run_training()

    reference_video_file = os.path.join(
        os.path.dirname(__file__), "reference_video_1_sample_v0.mp4"
    )
    print(f"reference_video_file: {reference_video_file}")
    final_validation_video_file = os.path.join(
        LOCAL_OUTPUT_DIR, "validation_step_900_inference_steps_50_video_0.mp4"
    )
    print(f"final_validation_video_file: {final_validation_video_file}")

    # Ensure both files exist
    assert os.path.exists(
        reference_video_file
    ), f"Reference video not found at {reference_video_file}"
    assert os.path.exists(
        final_validation_video_file
    ), f"Validation video not found at {final_validation_video_file}"

    # Compute SSIM
    mean_ssim, min_ssim, max_ssim = compute_video_ssim_torchvision(
        reference_video_file,
        final_validation_video_file,
        use_ms_ssim=True,  # Using MS-SSIM for better quality assessment
    )

    print("\n===== SSIM Results for Step 900 Validation =====")
    print(f"Mean MS-SSIM: {mean_ssim:.4f}")
    print(f"Min MS-SSIM: {min_ssim:.4f}")
    print(f"Max MS-SSIM: {max_ssim:.4f}")

    assert max_ssim > 0.5, f"Max SSIM is below 0.5: {max_ssim}"


if __name__ == "__main__":
    test_e2e_overfit_single_sample()
