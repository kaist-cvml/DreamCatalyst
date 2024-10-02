from dc_nerf.data.datamanagers.dc_datamanager import DCDataManagerConfig
from dc_nerf.data.datamanagers.dc_splat_datamanager import \
    DCSplatDataManagerConfig
from dc_nerf.data.dataparsers.dc_dataparser import DCDataParserConfig
from dc_nerf.engine.dc_trainer import DCTrainerConfig
from dc_nerf.models.dc_nerfacto import DCNerfactoModelConfig
from dc_nerf.models.dc_splatfacto import DCSplatfactoModelConfig
from dc_nerf.pipelines.dc_pipeline import DCPipelineConfig
from dc_nerf.pipelines.refinement_pipeline import RefinementPipelineConfig

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import \
    VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from dc.dc import DCConfig

nerfacto_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_eval_all_images=35000,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        experiment_name=None,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=DCNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    ),
    description="Nerfacto that can turn off the use of appearance embedding",
)
dc_method = MethodSpecification(
    config=DCTrainerConfig(
        method_name="dc",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=3000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=DCPipelineConfig(
            dc=DCConfig(src_prompt="", tgt_prompt="", guidance_scale=7.5),
            datamanager=DCDataManagerConfig(
                dataparser=DCDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=32,
            ),
            model=DCNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_appearance_embedding=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=100),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=100),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=1000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    ),
    description="DC-based NeRF editing method",
)

refinement_method = MethodSpecification(
    config=DCTrainerConfig(
        method_name="dc_refinement",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=RefinementPipelineConfig(
            dc=DCConfig(src_prompt="", tgt_prompt="", num_inference_steps=20, guidance_scale=15),
            skip_min_ratio=0.8,
            skip_max_ratio=0.9,
            datamanager=DCDataManagerConfig(
                dataparser=DCDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=32,
            ),
            model=DCNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_appearance_embedding=False,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, warmup_steps=1000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    ),
    description="Refinement Stage of DC",
)

dc_splat_method = MethodSpecification(
    config=DCTrainerConfig(
        method_name="dc_splat",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=3000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=DCPipelineConfig(
            dc=DCConfig(src_prompt="", tgt_prompt=""),
            datamanager=DCSplatDataManagerConfig(
                dataparser=DCDataParserConfig(),
                patch_size=32,
            ),
            model=DCSplatfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_downscales=0,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    warmup_steps=100,
                    max_steps=3000,
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, 
                    max_steps=3000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    ),
    description="DC-based 3D Gaussian Splat editing method",
)

dc_splat_refinement_method = MethodSpecification(
    config=DCTrainerConfig(
        method_name="dc_splat_refinement",
        steps_per_eval_batch=999999,
        steps_per_eval_image=999999,
        steps_per_eval_all_images=99999999,
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        load_scheduler=False,
        pipeline=RefinementPipelineConfig(
            dc=DCConfig(src_prompt="", tgt_prompt="", num_inference_steps=20, guidance_scale=15),
            skip_min_ratio=0.8,
            skip_max_ratio=0.9,
            datamanager=DCSplatDataManagerConfig(
                dataparser=DCDataParserConfig(),
                patch_size=32,
            ),
            model=DCSplatfactoModelConfig(
                num_downscales=0,
                stop_split_at=0, 
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    ),
    description="Refinement Stage of DC-Splat",
)
