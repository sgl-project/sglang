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

import os
import time
import glob
import json
import random
import numpy as np
import torch
from .loader_util import BaseDataset


class TextureDataset(BaseDataset):

    def __init__(
        self, json_path, num_view=6, image_size=512, lighting_suffix_pool=["light_PL", "light_AL", "light_ENVMAP"]
    ):
        self.data = list()
        self.num_view = num_view
        self.image_size = image_size
        self.lighting_suffix_pool = lighting_suffix_pool
        if isinstance(json_path, str):
            json_path = [json_path]
        for jp in json_path:
            with open(jp) as f:
                self.data.extend(json.load(f))
        print("============= length of dataset %d =============" % len(self.data))

    def __getitem__(self, index):
        try_sleep_interval = 20
        total_try_num = 100
        cnt = try_sleep_interval * total_try_num
        # try:
        images_ref = list()
        images_albedo = list()
        images_mr = list()
        images_normal = list()
        images_position = list()
        bg_white = [1.0, 1.0, 1.0]
        bg_black = [0.0, 0.0, 0.0]
        bg_gray = [127 / 255.0, 127 / 255.0, 127 / 255.0]
        dirx = self.data[index]

        condition_dict = {}

        # 6view
        fix_num_view = self.num_view
        available_views = []
        for ext in ["*_albedo.png", "*_albedo.jpg", "*_albedo.jpeg"]:
            available_views.extend(glob.glob(os.path.join(dirx, "render_tex", ext)))
        cond_images = (
            glob.glob(os.path.join(dirx, "render_cond", "*.png"))
            + glob.glob(os.path.join(dirx, "render_cond", "*.jpg"))
            + glob.glob(os.path.join(dirx, "render_cond", "*.jpeg"))
        )

        # 确保有足够的样本
        if len(available_views) < fix_num_view:
            print(
                f"Warning: Only {len(available_views)} views available, but {fix_num_view} requested."
                "Using all available views."
            )
            images_gen = available_views
        else:
            images_gen = random.sample(available_views, fix_num_view)

        if not cond_images:
            raise ValueError(f"No condition images found in {os.path.join(dirx, 'render_cond')}")
        ref_image_path = random.choice(cond_images)
        light_suffix = None
        for suffix in self.lighting_suffix_pool:
            if suffix in ref_image_path:
                light_suffix = suffix
                break
        if light_suffix is None:
            raise ValueError(f"light suffix not found in {ref_image_path}")
        ref_image_diff_light_path = random.choice(
            [
                ref_image_path.replace(light_suffix, tar_suffix)
                for tar_suffix in self.lighting_suffix_pool
                if tar_suffix != light_suffix
            ]
        )
        images_ref_paths = [ref_image_path, ref_image_diff_light_path]

        # Data aug
        bg_c_record = None
        for i, image_ref in enumerate(images_ref_paths):
            if random.random() < 0.6:
                bg_c = bg_gray
            else:
                if random.random() < 0.5:
                    bg_c = bg_black
                else:
                    bg_c = bg_white
            if i == 0:
                bg_c_record = bg_c
            image, alpha = self.load_image(image_ref, bg_c_record)
            image = self.augment_image(image, bg_c_record).float()
            images_ref.append(image)
        condition_dict["images_cond"] = torch.stack(images_ref, dim=0).float()

        for i, image_gen in enumerate(images_gen):
            images_albedo.append(self.augment_image(self.load_image(image_gen, bg_gray)[0], bg_gray))
            images_mr.append(
                self.augment_image(self.load_image(image_gen.replace("_albedo", "_mr"), bg_gray)[0], bg_gray)
            )
            images_normal.append(
                self.augment_image(self.load_image(image_gen.replace("_albedo", "_normal"), bg_gray)[0], bg_gray)
            )
            images_position.append(
                self.augment_image(self.load_image(image_gen.replace("_albedo", "_pos"), bg_gray)[0], bg_gray)
            )

        condition_dict["images_albedo"] = torch.stack(images_albedo, dim=0).float()
        condition_dict["images_mr"] = torch.stack(images_mr, dim=0).float()
        condition_dict["images_normal"] = torch.stack(images_normal, dim=0).float()
        condition_dict["images_position"] = torch.stack(images_position, dim=0).float()
        condition_dict["name"] = dirx  # .replace('/', '_')
        return condition_dict  # (N, 3, H, W)

        # except Exception as e:
        #     print(e, self.data[index])
        #     # exit()


if __name__ == "__main__":
    dataset = TextureDataset(json_path=["../../../train_examples/examples.json"])
    print("images_cond", dataset[0]["images_cond"].shape)
    print("images_albedo", dataset[0]["images_albedo"].shape)
    print("images_mr", dataset[0]["images_mr"].shape)
    print("images_normal", dataset[0]["images_normal"].shape)
    print("images_position", dataset[0]["images_position"].shape)
    print("name", dataset[0]["name"])
