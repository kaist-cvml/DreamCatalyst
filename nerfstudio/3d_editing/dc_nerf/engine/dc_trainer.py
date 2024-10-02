import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState


@dataclass
class DCTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: DCTrainer)


class DCTrainer(Trainer):
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1):
        super().__init__(config, local_rank, world_size)
        self.base_dir: Path = config.get_base_dir()

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            base_dir=self.base_dir,
            wandb_enabled=self.config.is_wandb_enabled(),
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None

        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )
        # save images after training.
        self.callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                func=self.pipeline.get_average_eval_image_metrics,
                kwargs={"output_path": self.base_dir / "eval_outputs"},
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)
