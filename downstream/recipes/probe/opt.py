import torch.optim as optim

from run_utils.callbacks.base import (AccumulateRawOutput, PeriodicSaver,
                                      ProcessAccumulatedRawOutput,
                                      ScalarMovingAverage, ScheduleLr, TrackLr,
                                      VisualizeOutput, TriggerEngine)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import (proc_cum_step_output, train_step, valid_step)

def get_config(
        train_loader_list, 
        infer_loader_list,
        loader_kwargs={},
        model_kwargs={},
        optimizer_kwargs={},
        **kwargs):

    config = {
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        'phase_list': [
            {
                'run_info': {
                    # may need more dynamic for each network
                    'net': {
                        'desc': lambda: create_model(**model_kwargs),
                        'optimizer': [
                            optim.Adam,
                            {
                                # should match keyword for parameters
                                # within the optimizer
                                'lr': 1.0e-4,  # initial learning rate,
                                'betas': (0.9, 0.999),
                                # 'weight_decay': 1.0e-5,
                            },
                        ],
                        # learning rate scheduler
                        'lr_scheduler': (
                            lambda opt, n_iter:
                                optim.lr_scheduler.StepLR(opt, 10000)),
                        # 'lr_scheduler': lambda opt, n_iter: \
                        #     optim.lr_scheduler.CosineAnnealingLR(opt, 50),

                        'extra_info': {
                        },

                        # path to load, -1 to auto load checkpoint
                        # from previous phase,
                        # None to start from scratch
                        'pretrained': None,
                    },
                },
                'target_info': {
                    'gen': (None, {}),
                    'viz': (None, {})
                },

                'loader': loader_kwargs,

                'nr_epochs': 50,
            },
        ],

        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        'run_engine': {
            'train': {
                'loader': train_loader_list,
                'run_step': train_step,
                'reset_per_run': False,

                # callbacks are run according to the list order of the event
                'callbacks': {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(),
                        ScalarMovingAverage(alpha=0.999),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        LoggingEpochOutput(),
                        TriggerEngine('infer'),
                        ScheduleLr(),
                    ]
                },
            },       
            'infer': {

                'loader': infer_loader_list,
                'run_step': valid_step,
                # * to stop aggregating output etc. from last run
                'reset_per_run': True,

                # callbacks are run according tothe list order of the event
                'callbacks': {
                    Events.STEP_COMPLETED: [
                        AccumulateRawOutput(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda a, b: proc_cum_step_output(a, b)),
                        LoggingEpochOutput(),
                    ]
                },
            },
        },
    }

    return config
