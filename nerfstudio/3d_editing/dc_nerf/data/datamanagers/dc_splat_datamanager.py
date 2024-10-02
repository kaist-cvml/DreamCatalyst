from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Type, cast

import torch
from rich.progress import Console

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager, FullImageDatamanagerConfig)
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate

CONSOLE = Console(width=120)


@dataclass
class DCSplatDataManagerConfig(FullImageDatamanagerConfig):
    """Configuration for the InstructNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: DCSplatDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

    train_num_rays_per_batch = 16384
    train_num_images_to_sample_from = -1
    train_num_times_to_repeat_images = -1

    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)


class DCSplatDataManager(FullImageDatamanager):
    """Data manager for InstructNeRF2NeRF."""

    config: DCSplatDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        self.image_batch = defaultdict(list)
        for i in range(len(self.cached_train)):
            for k, v in self.cached_train[i].items():
                self.image_batch[k].append(v)
        self.image_batch["image"] = torch.stack(self.image_batch["image"], 0)
        self.image_batch["image_idx"] = torch.tensor(self.image_batch["image_idx"])

        self.original_image_batch = {}
        self.original_image_batch["image"] = self.image_batch["image"].clone()
        self.original_image_batch["image_idx"] = self.image_batch["image_idx"].clone()

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        self.train_count += 1
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])
        # replace image
        data["image"] = self.image_batch["image"][image_idx].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data
