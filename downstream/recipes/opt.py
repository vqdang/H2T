from abc import ABC
from typing import List

import torch.optim as optim
from engine.callbacks.base import (
    AccumulateRawOutput,
    BaseCallbacks,
    PeriodicSaver,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    TriggerEngine,
)
from engine.callbacks.logging import LoggingOutput
from engine.engine import Events

from .base import ABCRecipe


def get_architecture(model_code: str, **kwargs):
    if model_code == "transformer-1":
        from models.mil.transformer import Transformer as Arch
    elif model_code == "transformer-2":
        from models.mil.transformer import Transformer as Arch
    elif model_code == "clam":
        from models.mil.clam import CLAM_SB as Arch
    elif model_code == "linear-probe":
        from models.probe import Probe as Arch
    elif model_code == "cnn-probe":
        from models.probe import Probe as Arch
    else:
        assert False
    return Arch(**kwargs)


class ProcessAccumulatedRawOutput(BaseCallbacks):
    def __init__(self, proc_func, per_n_epoch=1):
        # TODO: allow dynamically attach specific procesing for `type`
        super().__init__()
        self.per_n_epoch = per_n_epoch
        self.proc_func = proc_func

    def run(self, state, event):
        current_epoch = state.curr_epoch
        # if current_epoch % self.per_n_epoch != 0: return
        raw_data = state.epoch_accumulated_output
        # TODO: allow full access ?
        track_dict = self.proc_func(state.loader_name, raw_data)
        # update global shared states
        state.tracked_step_output = track_dict
        return


class ABCConfig(ABC):
    def __init__(
        self,
        recipe: ABCRecipe,
        train_loaders,
        infer_loaders,
        model_code: str,
        model_kwargs={},
        loader_kwargs={},
        optimizer_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.recipe = recipe
        self.train_loaders = train_loaders
        self.infer_loaders = infer_loaders
        self.loader_kwargs = loader_kwargs
        self.model_code = model_code
        self.model_kwargs = model_kwargs
        self.optimizer_kwargs = optimizer_kwargs

    def phases(self):
        phase_configs = [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: get_architecture(
                            self.model_code, **self.model_kwargs
                        ),
                        "optimizer": [
                            optim.Adam,
                            {
                                # should match keyword for parameters
                                # within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                                # 'weight_decay': 1.0e-5,
                            },
                        ],
                        # learning rate scheduler
                        "lr_scheduler": (
                            lambda opt, n_iter: optim.lr_scheduler.StepLR(opt, 10000)
                        ),
                        # 'lr_scheduler': lambda opt, n_iter: \
                        #     optim.lr_scheduler.CosineAnnealingLR(opt, 50),
                        "extra_info": {},
                        # path to load, -1 to auto load checkpoint
                        # from previous phase,
                        # None to start from scratch
                        "pretrained": None,
                    },
                },
                "target_info": {"gen": (None, {}), "viz": (None, {})},
                "loader": self.loader_kwargs,
                "nr_epochs": 50,
            },
        ]
        return phase_configs

    def engines(self):
        config = {
            "train": {
                "loader": self.train_loaders,
                "run_step": self.recipe.train_step,
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(),
                        ScalarMovingAverage(alpha=0.999),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        LoggingOutput(),
                        TriggerEngine("infer"),
                        ScheduleLr(),
                    ],
                },
            },
            "infer": {
                "loader": self.infer_loaders,
                "run_step": self.recipe.valid_step,
                # * to stop aggregating output etc. from last run
                "reset_per_run": True,
                # callbacks are run according tothe list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda a, b: self.recipe.process_accumulated_step_output(
                                a, b
                            )
                        ),
                        LoggingOutput(),
                    ],
                },
            },
        }
        return config

    @staticmethod
    def config(paramset, dataset_names: List[str], num_types: int = 2):
        """
        Args:
            dataset_names: A list contain names of dataset to be used
                by the training process. These names are used for initializing
                the dataloader of each engine defined in the coressponding
                configuration class.

        """
        loader_kwargs = {
            k: paramset["loader_kwargs"]["train"]
            if "train" in k
            else paramset["loader_kwargs"]["infer"]
            for k in dataset_names
        }
        paramset["loader_kwargs"] = loader_kwargs
        paramset["model_kwargs"]["num_types"] = num_types

        train_loaders = [v for v in dataset_names if "train" in v]
        infer_loaders = [v for v in dataset_names if not ("train" in v)]

        recipe_info = paramset["metadata"]
        if recipe_info["option_name"] == "linear":
            OptionClass = ABCConfig
        else:
            OptionClass = ABCConfig

        running_step_recipe = ABCRecipe.recipe(recipe_info["architecture_name"])
        config_ = OptionClass(
            running_step_recipe,
            train_loaders,
            infer_loaders,
            model_code=recipe_info["architecture_name"],
            model_kwargs=paramset["model_kwargs"],
            loader_kwargs=paramset["loader_kwargs"],
            optimizer_kwargs=paramset["optimizer_kwargs"],
        )
        config_ = {"phase_list": config_.phases(), "run_engine": config_.engines()}
        return config_
