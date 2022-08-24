
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import scipy
from typing import Union, Mapping, Callable

class ABCFeatures(ABC):
    """
    """
    def __init__(
            self,
            compute_basic_stats: Callable[[Union[list, np.ndarray]],
                                          Mapping[str, str]] = None
            ):

        if compute_basic_stats is not None:
            self._compute_basic_stats = compute_basic_stats
        return

    def _compute_basic_stats(
                self,
                val_list: Union[list, np.ndarray]
            ) -> Mapping[str, float]:

        val_list = np.array(val_list)
        stat_dict = OrderedDict()
        stat_dict['Max'] = np.max(val_list)
        stat_dict['Min'] = np.min(val_list)
        stat_dict['Mean'] = np.mean(val_list)
        stat_dict['Kurtosis'] = scipy.stats.skew(val_list)
        stat_dict['Skewness'] = scipy.stats.kurtosis(val_list)
        for percent in np.arange(0.1, 1.0, 0.1):
            sub_val = np.quantile(val_list, percent)
            stat_dict[f'Percentile-{percent:0.1f}'] = sub_val
            stat_dict[f'%Instances > Percentile-{percent:0.1f}'] = (
                np.mean(val_list > sub_val)
            )
        return stat_dict
