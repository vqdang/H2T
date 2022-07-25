import torch.optim as optim
from run_utils.callbacks.base import (AccumulateRawOutput, PeriodicSaver,
                                      ProcessAccumulatedEpochOutput,
                                      ScalarMovingAverage, ScheduleLr, TrackLr,
                                      TriggerEngine, VisualizeOutput)
from run_utils.callbacks.logging import LoggingGradient, LoggingOutput
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import (ProcStepRawOutput, proc_cum_epoch_output, train_step,
                       valid_step)


# TODO: training config only ?
# TODO: switch all to function name String for all option
def get_config(
        train_loader_list, 
        infer_loader_list,
        loader_kwargs=None, 
        model_kwargs={},
        optimizer_kwargs=None,
        **kwargs):

    config = {
        #------------------------------------------------------------------

        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        'phase_list': [
            {
                'run_info': {
                    # may need more dynamic for each network
                    'net': {
                        'desc': lambda: create_model(freeze_encoder=True, **model_kwargs),
                        'optimizer': [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                'lr': 1.0e-4,  # initial learning rate,
                                'betas': (0.9, 0.999),
                                # 'weight_decay' : 1.0e-5,
                            },
                        ],
                        # learning rate scheduler
                        'lr_scheduler': lambda opt, n_iter: \
                                optim.lr_scheduler.StepLR(opt, 10000),    
                        # 'lr_scheduler': lambda opt, n_iter: \
                        #     optim.lr_scheduler.CosineAnnealingLR(opt, 50),
                        
                        'extra_info' : {
                        },

                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        'pretrained': None,
                    },
                },
                'target_info': { 
                    'gen' : (None, {}), 
                    'viz' : (None, {})
                },

                'loader' : loader_kwargs,

                'nr_epochs': 20,
            },
            {
                'run_info': {
                    # may need more dynamic for each network
                    'net': {
                        'desc': lambda: create_model(freeze_encoder=False, **model_kwargs),
                        'optimizer': [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                'lr': 1.0e-4,  # initial learning rate,
                                'betas': (0.9, 0.999),
                                # 'weight_decay' : 1.0e-5,
                            },
                        ],
                        # learning rate scheduler
                        'lr_scheduler': lambda opt, n_iter: \
                                optim.lr_scheduler.StepLR(opt, 10000),    
                        # 'lr_scheduler': lambda opt, n_iter: \
                        #     optim.lr_scheduler.CosineAnnealingLR(opt, 50),
                        
                        'extra_info' : {
                        },

                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        'pretrained': -1,
                    },
                },
                'target_info': { 
                    'gen' : (None, {}), 
                    'viz' : (None, {})
                },

                'loader' : loader_kwargs,

                'nr_epochs': 50,
            },
        ],

        #------------------------------------------------------------------

        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        'run_engine': {
            'train': {
                # TODO: align here, file path or what? what about CV?
                'loader'     : train_loader_list,
                'run_step'   : train_step, # TODO: function name or function variable ?
                'reset_per_run' : False,

                # callbacks are run according to the list order of the event
                'callbacks': {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(),
                        ScalarMovingAverage(alpha=0.999),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        LoggingOutput(),
                        TriggerEngine('infer'),
                        ScheduleLr(),
                    ]
                },
            },       
            'infer' : {

                'loader'    : infer_loader_list,
                'run_step'   : valid_step,
                'reset_per_run' : True, # * to stop aggregating output etc. from last run
                
                # callbacks are run according to the list order of the event            
                'callbacks' : {
                    Events.STEP_COMPLETED  : [
                        ProcStepRawOutput(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedEpochOutput(lambda a, b : proc_cum_epoch_output(a, b)),
                        LoggingOutput(),
                    ]
                },
            },
        },
    }

    return config
