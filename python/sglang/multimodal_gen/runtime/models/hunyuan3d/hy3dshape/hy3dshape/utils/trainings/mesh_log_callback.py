# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import json
import math
import os
from typing import Tuple, Generic, Dict, List, Union, Optional

import trimesh
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.utils.trainings.mesh import MeshOutput
from hy3dshape.utils.visualizers import html_util
from hy3dshape.utils.visualizers.pythreejs_viewer import PyThreeJSViewer


class ImageConditionalASLDiffuserLogger(Callback): 
    def __init__(self,
                 step_frequency: int,
                 num_samples: int = 1,
                 mean: Optional[Union[List[float], Tuple[float]]] = None,
                 std: Optional[Union[List[float], Tuple[float]]] = None,
                 bounds: Union[List[float], Tuple[float]] = (-1.1, -1.1, -1.1, 1.1, 1.1, 1.1),
                 **kwargs) -> None:

        super().__init__()
        self.bbox_size = np.array(bounds[3:6]) - np.array(bounds[0:3])

        if mean is not None:
            mean = np.asarray(mean)

        if std is not None:
            std = np.asarray(std)

        self.mean = mean
        self.std = std

        self.step_freq = step_frequency
        self.num_samples = num_samples
        self.has_train_logged = False
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }

        self.viewer = PyThreeJSViewer(settings={}, render_mode="WEBSITE")

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        # raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    def log_local(self,
                  outputs: List[List['Latent2MeshOutput']],
                  images: Union[np.ndarray, List[np.ndarray]],
                  description: List[str],
                  keys: List[str],
                  save_dir: str, split: str,
                  global_step: int, current_epoch: int, batch_idx: int,
                  prog_bar: bool = False,
                  multi_views=None,  # yf ...
                  ) -> None:

        folder = "gs-{:010}_e-{:06}_b-{:06}".format(global_step, current_epoch, batch_idx)
        visual_dir = os.path.join(save_dir, "visuals", split, folder)
        os.makedirs(visual_dir, exist_ok=True)

        num_samples = len(images)
        
        for i in range(num_samples):
            key_i = keys[i]
            image_i = self.denormalize_image(images[i])
            shape_tag_i = description[i]

            for j in range(1):
                mesh = outputs[j][i]
                if mesh is None:
                    continue

                mesh_v = mesh.mesh_v.copy()
                mesh_v[:, 0] += j * np.max(self.bbox_size)
                self.viewer.add_mesh(mesh_v, mesh.mesh_f)

            image_tag = html_util.to_image_embed_tag(image_i)
            mesh_tag = self.viewer.to_html(html_frame=False)

            table_tag = f"""
            <table border = "1">
                <caption> {shape_tag_i} - {key_i} </caption>
                <caption> Input Image | Generated Mesh </caption>
                <tr>
                    <td>{image_tag}</td>
                    <td>{mesh_tag}</td>
                </tr>
            </table>
            """

            if multi_views is not None:
                multi_views_i = self.make_grid(multi_views[i])
                views_tag = html_util.to_image_embed_tag(self.denormalize_image(multi_views_i))
                table_tag = f"""
                <table border = "1">
                    <caption> {shape_tag_i} - {key_i} </caption>
                    <caption> Input Image | Generated Mesh </caption>
                    <tr>
                        <td>{image_tag}</td>
                        <td>{views_tag}</td>
                        <td>{mesh_tag}</td>
                    </tr>
                </table>
                """

            html_frame = html_util.to_html_frame(table_tag)
            if len(key_i) > 100:
                key_i = key_i[:100]
            with open(os.path.join(visual_dir, f"{key_i}.html"), "w") as writer:
                writer.write(html_frame)

            self.viewer.reset()

    def log_sample(self,
                   pl_module: pl.LightningModule,
                   batch: Dict[str, torch.FloatTensor],
                   batch_idx: int,
                   split: str = "train") -> None:
        """

        Args:
            pl_module:
            batch (dict): the batch sample information, and it contains:
                 - surface (torch.FloatTensor):
                 - image (torch.FloatTensor):
            batch_idx (int):
            split (str):

        Returns:

        """

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        batch_size = len(batch["surface"])
        replace = batch_size < self.num_samples
        ids = np.random.choice(batch_size, self.num_samples, replace=replace)

        with torch.no_grad():
            # run text to mesh
            # keys = [batch["__key__"][i] for i in ids]
            keys = [f'key_{i}' for i in ids]
            # texts = [batch["text"][i] for i in ids]
            texts = [f'text_{i}'for i in ids]
            # description = [batch["description"][i] for i in ids]
            description = [f'desc_{i}' for i in ids]
            images = batch["image"][ids]
            mask_input = batch["mask"][ids] if 'mask' in batch else None
            sample_batch = {
                "__key__": keys,
                "image": images,
                'text': texts,
                'mask': mask_input,
            }

            # if 'cam_parm' in batch:
            #     sample_batch['cam_parm'] = batch['cam_parm'][ids]

            # if 'multi_views' in batch:  # yf ...
            #     sample_batch['multi_views'] = batch['multi_views'][ids]

            outputs = pl_module.sample(
                batch=sample_batch,
                output_type='latents2mesh'
            )

            images = images.cpu().float().numpy()
            # images = self.denormalize_image(images)
            # images = np.transpose(images, (0, 2, 3, 1))
            # images = ((images + 1) / 2 * 255).astype(np.uint8)

        self.log_local(outputs, images, description, keys, pl_module.logger.save_dir, split,
                       pl_module.global_step, pl_module.current_epoch, batch_idx, prog_bar=False,
                       multi_views=sample_batch.get('multi_views'))

        if is_train: pl_module.train()

    def make_grid(self, images):  # return (3,h,w) in (0,1) ...
        images_resized = []
        for img in images:
            img_resized = torchvision.transforms.functional.resize(img, (320, 320))
            images_resized.append(img_resized)
        image = torchvision.utils.make_grid(images_resized, nrow=2, padding=5, pad_value=255)

        image = image.cpu().numpy()
        #       image = np.transpose(image, (1, 2, 0))
        #       image = (image * 255).astype(np.uint8)

        return image

    def check_frequency(self, step: int) -> bool:
        if step % self.step_freq == 0:
            return True
        return False

    def on_train_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                           outputs: Generic, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> None:

        if (self.check_frequency(pl_module.global_step) and  # batch_idx % self.batch_freq == 0
            hasattr(pl_module, "sample") and
            callable(pl_module.sample) and
            self.num_samples > 0):
            self.log_sample(pl_module, batch, batch_idx, split="train")
            self.has_train_logged = True

    def on_validation_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                                outputs: Generic, batch: Dict[str, torch.FloatTensor],
                                dataloader_idx: int, batch_idx: int) -> None:

        if self.has_train_logged:
            self.log_sample(pl_module, batch, batch_idx, split="val")
            self.has_train_logged = False

    def denormalize_image(self, image):
        """

        Args:
            image (np.ndarray): [3, h, w]

        Returns:
            image (np.ndarray): [h, w, 3], np.uint8, [0, 255].
        """
        # image = np.transpose(image, (0, 2, 3, 1))
        image = np.transpose(image, (1, 2, 0))

        if self.std is not None:
            image = image * self.std

        if self.mean is not None:
            image = image + self.mean

        image = (image * 255).astype(np.uint8)

        return image


class ImageConditionalFixASLDiffuserLogger(Callback):
    def __init__(
        self,
        step_frequency: int,
        test_data_path: str,
        max_size: int = None,
        save_dir: str = 'infer',
        **kwargs,
    ) -> None:
        super().__init__()
        self.step_freq = step_frequency
        self.viewer = PyThreeJSViewer(settings={}, render_mode="WEBSITE")

        self.test_data_path = test_data_path
        with open(self.test_data_path, 'r') as f:
            data = json.load(f)
            self.file_list = data['file_list']
            self.file_folder = data['file_folder']
            if max_size is not None:
                self.file_list = self.file_list[:max_size]
        self.kwargs = kwargs
        self.save_dir = save_dir

    def on_train_batch_end(
        self,
        trainer: pl.trainer.Trainer,
        pl_module: pl.LightningModule,
        outputs: Generic,
        batch: Dict[str, torch.FloatTensor],
        batch_idx: int,
    ):
        if pl_module.global_step % self.step_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            folder_path = self.file_folder
            folder_name = os.path.basename(folder_path)
            folder = "gs-{:010}_e-{:06}_b-{:06}".format(pl_module.global_step, pl_module.current_epoch, batch_idx)
            visual_dir = os.path.join(pl_module.logger.save_dir, self.save_dir, folder, folder_name)
            os.makedirs(visual_dir, exist_ok=True)

            image_paths = self.file_list
            chunk_size = math.ceil(len(image_paths) / trainer.world_size)
            if pl_module.global_rank == trainer.world_size - 1:
                image_paths = image_paths[pl_module.global_rank * chunk_size:]
            else:
                image_paths = image_paths[pl_module.global_rank * chunk_size:(pl_module.global_rank + 1) * chunk_size]

            print(f'Rank{pl_module.global_rank}: processing {len(image_paths)}|{len(self.file_list)} images')
            for image_path in image_paths:
                if folder_path in image_path:
                    save_path = image_path.replace(folder_path, visual_dir)
                else:
                    save_path = os.path.join(visual_dir, os.path.basename(image_path))
                save_path = os.path.splitext(save_path)[0] + '.glb'

                if isinstance(image_path, str):
                    print(image_path)
                    
                with torch.no_grad():
                    mesh = pl_module.sample(batch={"image": image_path}, **self.kwargs)[0][0]
                    if isinstance(mesh, tuple) and len(mesh)==2:
                        mesh = export_to_trimesh(mesh)
                    elif isinstance(mesh, trimesh.Trimesh):
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        mesh.export(save_path)

            if is_train:
                pl_module.train()
