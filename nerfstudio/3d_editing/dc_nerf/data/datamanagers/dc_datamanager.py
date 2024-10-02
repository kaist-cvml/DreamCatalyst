from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator

CONSOLE = Console(width=120)


@dataclass
class DCDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: DCDataManager)


class DCDataManager(VanillaDataManager):
    config: DCDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch["image"] = self.image_batch["image"].clone()
        self.original_image_batch["image_idx"] = self.image_batch["image_idx"].clone()

    def next_train(self, step: int, use_original_image: bool = False) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None
        if use_original_image:
            batch = self.train_pixel_sampler.sample(self.original_image_batch)
        else:
            batch = self.train_pixel_sampler.sample(self.image_batch)
        # batch: dict of "image": [num_rays_per_batch, 3], "indices": [num_rays_per_batch, 3]
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
