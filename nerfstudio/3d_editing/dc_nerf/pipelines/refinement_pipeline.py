import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Literal, Optional, Type, Union
from dc_nerf.data.datamanagers.dc_splat_datamanager import DCSplatDataManagerConfig

import numpy as np
import torch

from dc_nerf.pipelines.base_pipeline import ModifiedVanillaPipeline
from dc_nerf.data.datamanagers.dc_datamanager import DCDataManagerConfig
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from dc.dc import DC, DCConfig, tensor_to_pil
from dc.utils import imageutil
from dc.utils.sysutil import clean_gpu


@dataclass
class RefinementPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: RefinementPipeline)

    datamanager: Union[DCDataManagerConfig, DCSplatDataManagerConfig] = DCDataManagerConfig()
    dc: DCConfig = DCConfig()
    dc_device: Optional[Union[torch.device, str]] = None

    skip_min_ratio: float = 0.8
    skip_max_ratio: float = 0.9

    log_step: int = 100
    edit_rate: int = 10
    edit_count: int = 1


class RefinementPipeline(ModifiedVanillaPipeline):
    config: RefinementPipelineConfig

    def __init__(
        self,
        config: RefinementPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler, **kwargs)

        # Construct DC
        self.dc_device = (
            torch.device(device) if self.config.dc_device is None else torch.device(self.config.dc_device)
        )
        self.config.dc.device = self.dc_device
        self.dc = DC(self.config.dc)

        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

    def get_current_rendering(self):
        current_spot = next(self.train_indices_order)
        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index : current_index + 1].to(
            self.device
        )
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]
        # delete to free up memory
        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, original_image, current_spot

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if step % self.config.edit_rate == 0:
            for i in range(self.config.edit_count):
                rendered_image, original_image, current_spot = self.get_current_rendering()
                input_img = original_image

                # with torch.no_grad():
                if True:
                    h, w = input_img.shape[2:]
                    l = min(h, w)
                    h = int(h * 512 / l)
                    w = int(w * 512 / l)

                    resized_img = torch.nn.functional.interpolate(input_img, size=(h, w), mode="bilinear")
                    latents = self.dc.encode_image(resized_img.to(self.dc_device))

                ## config ##
                x0 = latents
                num_inference_steps = self.dc.config.num_inference_steps
                min_step = int(num_inference_steps * self.config.skip_min_ratio)
                max_step = int(num_inference_steps * self.config.skip_max_ratio)
                skip = random.randint(min_step, max_step)

                edit_x0 = self.dc.run_sdedit(x0, skip=skip)
                edit_img = self.dc.decode_latent(edit_x0)

                if edit_img.size() != rendered_image.size():
                    edit_img = torch.nn.functional.interpolate(
                        edit_img, size=rendered_image.size()[2:], mode="bilinear"
                    )

                self.datamanager.image_batch["image"][current_spot] = edit_img.squeeze().permute(1, 2, 0)  # [H,W,3]

            if step % self.config.log_step == 0:
                with torch.no_grad():
                    rendered_img_pil = tensor_to_pil(rendered_image)
                    edit_img_pil = tensor_to_pil(edit_img)
                    rw, rh = float("inf"), float("inf")
                    for img in [rendered_img_pil, edit_img_pil]:
                        w, h = img.size
                        rw = min(w, rw)
                        rh = min(h, rh)
                    rendered_img_pil = rendered_img_pil.resize((rw, rh))
                    edit_img_pil = edit_img_pil.resize((rw, rh))
                    save_img_pil = imageutil.merge_images([rendered_img_pil, edit_img_pil])
                    save_img_pil.save(self.base_dir / f"logging/replace-out-{step}.png")

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict
