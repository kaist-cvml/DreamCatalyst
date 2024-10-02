import os
import math
import numpy as np
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from time import time
import typing
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import (CAMERA_MODEL_TO_TYPE, Cameras,
                                        CameraType)
from nerfstudio.data.datamanagers.base_datamanager import (DataManager,
                                                           VanillaDataManager)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.utils import profiler
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dc.utils.imageutil import images2gif


class ModifiedVanillaPipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        if kwargs.get("base_dir"):
            self.base_dir = kwargs.get("base_dir")
            (self.base_dir / "logging").mkdir(exist_ok=True, parents=True)

        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        if output_path is not None:
            (output_path / "images").mkdir(exist_ok=True, parents=True)

        if self.datamanager.dataparser.config.data.suffix == ".json":
            metadata_path = self.datamanager.dataparser.config.data
        else:
            metadata_path = self.datamanager.dataparser.config.data / "transforms.json"
        metadata_path = os.path.realpath(metadata_path)
        new_path = output_path / "transforms.json"
        os.system(f"cp {metadata_path} {new_path}")

        self.eval()
        metrics_dict_list = []

        # render_dataset contains all images including train set and eval set.
        self.render_dataset = self.datamanager.dataset_type(
            dataparser_outputs=self.get_all_dataparser_outputs(),
            scale_factor=self.datamanager.config.camera_res_scale_factor,
        )
        fixed_indices_all_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.render_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        ####
        num_images = len(fixed_indices_all_dataloader)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)

            gif_images = []
            gif_img_filenames = []
            for i, (camera, batch) in enumerate(fixed_indices_all_dataloader):
                image_filename = Path(self.render_dataset.image_filenames[i]).stem

                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    # raise NotImplementedError("Saving images is not implemented yet")
                    for key, val in images_dict.items():
                        img = Image.fromarray((val * 255).byte().cpu().numpy())
                        if key == "img":
                            img.save(output_path / f"images/{image_filename}.png")
                            gif_images.append(img)
                            gif_img_filenames.append(image_filename)


                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        
        # save gif
        if output_path is not None:
            h, w = gif_images[0].size
            l = max(h, w)
            if l > 512:
                h = int(h * 512 / l)
                w = int(w * 512 / l)
                gif_images = [x.resize((h,w)) for x in gif_images]
                gif_img_filenames, gif_images = zip(
                    *sorted(zip(gif_img_filenames, gif_images), key=lambda pair: pair[0]
                    )
                )
            images2gif(gif_images, output_path / f"images/animation.gif")

        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [metrics_dict[key] for metrics_dict in metrics_dict_list]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                )
        self.train()
        return metrics_dict

    def get_all_dataparser_outputs(self):
        split = "all"
        dataparser = self.datamanager.dataparser
        assert dataparser.config.data.exists()

        if dataparser.config.data.suffix == ".json":
            meta = load_from_json(dataparser.config.data)
            data_dir = dataparser.config.data.parent
        else:
            meta = load_from_json(dataparser.config.data / "transforms.json")
            data_dir = dataparser.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        if dataparser.config.sort_images_based_on_name:
            frames = meta["frames"]
            frames = sorted(frames, key=lambda x: x["file_path"])
            print(
                f"[*] Sorted frames"
            )  # frames[0]['file_path'], frames[1]['file_path'], frames[2]['file_path'])
        else:
            frames = meta["frames"]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = dataparser._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = dataparser._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = dataparser._get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(
            f"{split}_filenames" in meta for split in ("train", "val", "test")
        )
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(
                dataparser._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"]
            )
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {unmatched_filenames}."
                )

            indices = [
                i for i, path in enumerate(image_filenames) if path in split_filenames
            ]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(
                f"The dataset's list of filenames for split {split} is missing."
            )
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(image_filenames)
            num_train_images = math.ceil(num_images * dataparser.config.train_split_fraction)

            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            indices = i_all

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(
                f"[yellow] Dataset is overriding orientation method to {orientation_method}"
            )
        else:
            orientation_method = dataparser.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=dataparser.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if dataparser.config.auto_scale_poses:
            if float(torch.max(torch.abs(poses[:, :3, 3]))) != 0:
                scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= dataparser.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = dataparser.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = (
            float(meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert dataparser.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / dataparser.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames
                if len(depth_filenames) > 0
                else None,
                "depth_unit_scale_factor": dataparser.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs
