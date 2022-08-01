from collections import abc

import models.mil as MIL
import torch.optim as optim
from engine.callbacks.base import (AccumulateRawOutput, PeriodicSaver,
                                   ProcessAccumulatedRawOutput,
                                   ScalarMovingAverage, ScheduleLr, TrackLr,
                                   TriggerEngine)
from engine.callbacks.logging import LoggingEpochOutput
from engine.engine import Events

from .base import ABCRecipe


def get_architecture(model_code='clam', **kwargs):
    if model_code == 'transformer-1':
        return MIL.transformer.Transformer(**kwargs)
    elif model_code == 'transformer-2':
        return MIL.transformer.Transformer(**kwargs)
    elif model_code == 'clam':
        return MIL.clam.CLAM_SB(**kwargs)
    elif model_code == 'linear-probe':
        return MIL.clam.CLAM_SB(**kwargs)
    elif model_code == 'cnn-probe':
        from models.probe import Probe
        return Probe(**kwargs)
    else:
        assert False


class ABCConfig(abc):
    def __init__(
        self,
        recipe: ABCRecipe,
        train_loaders,
        infer_loaders,
        loader_kwargs={},
        model_kwargs={},
        optimizer_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.recipe = recipe
        self.train_loaders = train_loaders
        self.infer_loaders = infer_loaders
        self.loader_kwargs = loader_kwargs
        self.model_kwargs = model_kwargs
        self.optimizer_kwargs = optimizer_kwargs

    def phases(self):
        phase_configs = [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: get_architecture(**self.model_kwargs),
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
                        LoggingEpochOutput(),
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
                    Events.STEP_COMPLETED: [
                        AccumulateRawOutput(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda a, b: self.recipe.process_accumulated_step_output(a, b)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        }
        return config

    def config(self):
        config_ = {"phase_list": self.phases, "run_engine": self.engines}
        return config_
