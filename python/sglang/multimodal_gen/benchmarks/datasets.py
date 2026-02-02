import glob
import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    image_paths: Optional[List[str]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: Dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0


def is_dir_not_empty(path):
    return os.path.isdir(path) and bool(os.listdir(path))


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model
        self.items: List[Dict[str, Any]] = []

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_request(self, idx: int) -> RequestFuncInput:
        pass

    @abstractmethod
    def get_requests(self) -> List[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    I2V_DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"

    def __init__(self, args, api_url: str = "", model: str = ""):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        self.items = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        if self.args.task_name in ("text-to-video", "text-to-image", "video-to-video"):
            return self._load_t2v_prompts()
        elif self.args.task_name in ("image-to-video", "image-to-image"):
            return self._load_i2v_data()
        else:
            raise ValueError(
                f"Illegal task name is found in VBenchDataset {self.args.task_name}"
            )

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> List[Dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                logger.info(f"Downloading VBench T2V prompts to {path}...")
                try:
                    self._download_file(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    logger.info(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> Optional[str]:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")
        crop_dir = os.path.join(vbench_i2v_dir, "data", "crop")
        origin_dir = os.path.join(vbench_i2v_dir, "data", "origin")

        if (
            os.path.exists(info_json_path)
            and is_dir_not_empty(crop_dir)
            and is_dir_not_empty(origin_dir)
        ):
            return vbench_i2v_dir

        logger.info(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            logger.info("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")
            missing_packages = re.findall(r"(\S+): command not found", result.stderr)
            if missing_packages:
                missing_packages = list(set(missing_packages))
                package_list = ", ".join(f"'{cmd}'" for cmd in missing_packages)
                raise RuntimeError(
                    f"Download script failed because the following commands are not installed: {package_list}.\n"
                    "Please install them (e.g., on Ubuntu: `sudo apt install ...`) and try again."
                )
            logger.info(
                f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}"
            )
        except Exception as e:
            logger.info(f"Failed to download VBench I2V dataset: {e}")
            logger.info("Please manually download following instructions at:")
            logger.info(
                "https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download"
            )
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load I2V data from i2v-bench-info.json format."""
        with open(json_path, "r") as f:
            items = json.load(f)

        base_dir = os.path.dirname(
            os.path.dirname(json_path)
        )  # Go up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        data = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                logger.warning(f"Image not found: {img_path}")

        logger.info(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> List[Dict[str, Any]]:
        """Scan directory for image files."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))

            # Also check in data/origin subdirectory
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [
            {"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f}
            for f in files
        ]

    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        """Create dummy data with a placeholder image in cache directory."""
        logger.info("No I2V data found. Using dummy placeholders.")

        dummy_image = os.path.join(self.cache_dir, "dummy_image.jpg")
        if not os.path.exists(dummy_image):
            try:
                from PIL import Image

                os.makedirs(self.cache_dir, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="red")
                img.save(dummy_image)
                logger.info(f"Created dummy image at {dummy_image}")
            except ImportError:
                logger.info("PIL not installed, cannot create dummy image.")
                return []

        return [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

    def _load_i2v_data(self) -> List[Dict[str, Any]]:
        """Load I2V data from VBench I2V dataset or user-provided path."""
        path = self.args.dataset_path
        # Auto-download if no path provided
        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        # Try to load from i2v-bench-info.json
        info_json_candidates = [
            os.path.join(path, "data", "i2v-bench-info.json"),
            path if path.endswith(".json") else None,
        ]

        for json_path in info_json_candidates:
            if json_path and os.path.exists(json_path):
                try:
                    return self._resize_data(self._load_from_i2v_json(json_path))
                except Exception as e:
                    logger.info(f"Failed to load {json_path}: {e}")

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        # Last resort: dummy data
        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resize data to match num_prompts."""
        if not self.args.num_prompts:
            return data

        if len(data) < self.args.num_prompts:
            factor = (self.args.num_prompts // len(data)) + 1
            data = data * factor

        return data[: self.args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self[i] for i in indices]

    def get_request(self, idx: int) -> RequestFuncInput:
        item = self[idx]
        assert (
            len(self.api_url) > 0
        ), "API URL must be provided for generating requests."
        assert len(self.model) > 0, "Model must be provided for generating requests."

        return RequestFuncInput(
            prompt=item.get("prompt", ""),
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            fps=self.args.fps,
            image_paths=[item["image_path"]] if "image_path" in item else None,
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self.get_request(i) for i in range(len(self))]


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str = "", model: str = ""):
        self.args = args
        self.api_url = api_url
        self.model = model
        self.num_prompts = args.num_prompts or 100

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prompt": f"Random prompt {idx} for benchmarking diffusion models",
            "width": self.args.width,
            "height": self.args.height,
            "num_frames": self.args.num_frames,
            "fps": self.args.fps,
        }

    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self[i] for i in indices]

    def get_request(self, idx: int) -> RequestFuncInput:
        assert (
            len(self.api_url) > 0
        ), "API URL must be provided for generating requests."
        assert len(self.model) > 0, "Model must be provided for generating requests."

        return RequestFuncInput(
            api_url=self.api_url,
            model=self.model,
            **self[idx],
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self.get_request(i) for i in range(len(self))]
