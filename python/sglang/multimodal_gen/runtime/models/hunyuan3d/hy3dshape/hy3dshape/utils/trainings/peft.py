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

import os
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf, ListConfig

class PeftSaveCallback(Callback):
    def __init__(self, peft_model, save_dir: str, save_every_n_steps: int = None):
        super().__init__()
        self.peft_model = peft_model
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def recursive_convert(self, obj):
        from omegaconf import OmegaConf, ListConfig
        if isinstance(obj, (OmegaConf, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: self.recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_convert(i) for i in obj]
        elif isinstance(obj, type):
            # 避免修改类对象
            return obj
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_value in vars(obj).items():
                setattr(obj, attr_name, self.recursive_convert(attr_value))
            return obj
        else:
            return obj

    # def recursive_convert(self, obj):
    #     if isinstance(obj, (OmegaConf, ListConfig)):
    #         return OmegaConf.to_container(obj, resolve=True)
    #     elif isinstance(obj, dict):
    #         return {k: self.recursive_convert(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):
    #         return [self.recursive_convert(i) for i in obj]
    #     elif hasattr(obj, '__dict__'):
    #         for attr_name, attr_value in vars(obj).items():
    #             setattr(obj, attr_name, self.recursive_convert(attr_value))
    #         return obj
    #     else:
    #         return obj

    def _convert_peft_config(self):
        pc = self.peft_model.peft_config
        self.peft_model.peft_config = self.recursive_convert(pc)

    def on_train_epoch_end(self, trainer, pl_module):
        self._convert_peft_config()
        save_path = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch}")
        self.peft_model.save_pretrained(save_path)
        print(f"[PeftSaveCallback] Saved LoRA weights to {save_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.save_every_n_steps is not None:
            global_step = trainer.global_step
            if global_step % self.save_every_n_steps == 0 and global_step > 0:
                self._convert_peft_config()
                save_path = os.path.join(self.save_dir, f"step_{global_step}")
                self.peft_model.save_pretrained(save_path)
                print(f"[PeftSaveCallback] Saved LoRA weights to {save_path}")
