"""
Code borrowed from nerfstudio==1.0.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class DCSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: DCSplatfactoModel)


class DCSplatfactoModel(SplatfactoModel):

    """
    differentiable version of `get_outputs_for_camera`
    """

    def diff_get_outputs_for_camera(
        self, camera: Cameras, obb_box: Optional[OrientedBox] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        img = predicted_rgb
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"combined": combined_rgb, "img": img}

        return metrics_dict, images_dict
