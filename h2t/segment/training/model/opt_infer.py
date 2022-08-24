import torch.optim as optim

from run_utils.callbacks.base import (AccumulateRawOutput, PeriodicSaver,
                                      ProcessAccumulatedRawOutput,
                                      ScalarMovingAverage, ScheduleLr, TrackLr,
                                      VisualizeOutput, TriggerEngine)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import (proc_valid_step_output, train_step, valid_step)

# TODO: training config only ?
# TODO: switch all to function name String for all option
def get_config(infer_loader_list, loader_kwargs, model_kwargs, 
               pretrained=None, save_path=None):
    config = {
        #------------------------------------------------------------------

        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        'phase_list': [
            {
                'run_info': {
                    # may need more dynamic for each network
                    'net': {
                        'desc': lambda: create_model(**model_kwargs),
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        'pretrained': pretrained,

                        'extra_info' : {
                        },
                    },
                },
                'target_info': { 
                    'gen' : (None, {}), 
                    'viz' : (None, {})
                },

                'loader' : loader_kwargs,
                'nr_epochs': 1,
            },
        ],

        #------------------------------------------------------------------

        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        'run_engine': {   
            'infer' : {
                'loader'   : infer_loader_list,
                'run_step' : valid_step,
                'reset_per_run' : True, # * to stop aggregating output etc. from last run
                
                # callbacks are run according to the list order of the event            
                'callbacks' : {
                    Events.STEP_COMPLETED  : [
                        AccumulateRawOutput(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda a, b : 
                                proc_valid_step_output(a, b, save_path=save_path)),
                        LoggingEpochOutput(),
                    ]
                },
            },
        },
    }

    return config
