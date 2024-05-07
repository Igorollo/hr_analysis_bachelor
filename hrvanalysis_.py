from typing import Tuple
from typing import List
import pandas as pd
import numpy as np

# Static name for methods params
MALIK_RULE = "malik"
KARLSSON_RULE = "karlsson"
KAMATH_RULE = "kamath"
ACAR_RULE = "acar"
CUSTOM_RULE = "custom"

def remove_outliers(rr_intervals: List[float], verbose: bool = True, low_rri: int = 300,
                    high_rri: int = 2000) -> list:
    """
    Function that replace RR-interval outlier by nan.

    Parameters
    ---------
    rr_intervals : list
        raw signal extracted.
    low_rri : int
        lowest RrInterval to be considered plausible.
    high_rri : int
        highest RrInterval to be considered plausible.
    verbose : bool
        Print information about deleted outliers.

    Returns
    ---------
    rr_intervals_cleaned : list
        list of RR-intervals without outliers

    References
    ----------
    .. [1] O. Inbar, A. Oten, M. Scheinowitz, A. Rotstein, R. Dlin, R.Casaburi. Normal \
    cardiopulmonary responses during incremental exercise in 20-70-yr-old men.

    .. [2] W. C. Miller, J. P. Wallace, K. E. Eggert. Predicting max HR and the HR-VO2 relationship\
    for exercise prescription in obesity.

    .. [3] H. Tanaka, K. D. Monahan, D. R. Seals. Age-predictedmaximal heart rate revisited.

    .. [4] M. Gulati, L. J. Shaw, R. A. Thisted, H. R. Black, C. N. B.Merz, M. F. Arnsdorf. Heart \
    rate response to exercise stress testing in asymptomatic women.
    """

    # Conversion RrInterval to Heart rate ==> rri (ms) =  1000 / (bpm / 60)
    # rri 2000 => bpm 30 / rri 300 => bpm 200
    
    rr_intervals_cleaned_ = []
    for rri in rr_intervals:
        if high_rri >= rri >= low_rri:
            rr_intervals_cleaned_.append(rri)
        else:
            rr_intervals_cleaned_.append(np.nan)
    rr_intervals_cleaned = rr_intervals_cleaned_
    if verbose:
        outliers_list = []
        for rri in rr_intervals:
            if high_rri >= rri >= low_rri:
                pass
            else:
                outliers_list.append(rri)

        nan_count = sum(np.isnan(rr_intervals_cleaned))
        if nan_count == 0:
            print("{} outlier(s) have been deleted.".format(nan_count))
        else:
            print("{} outlier(s) have been deleted.".format(nan_count))
            print("The outlier(s) value(s) are : {}".format(outliers_list))
    return rr_intervals_cleaned

def remove_ectopic_beats(rr_intervals: List[float], method: str = "malik",
                         custom_removing_rule: float = 0.2, verbose: bool = True) -> list:
    """
    RR-intervals differing by more than the removing_rule from the one proceeding it are removed.

    Parameters
    ---------
    rr_intervals : list
        list of RR-intervals
    method : str
        method to use to clean outlier. malik, kamath, karlsson, acar or custom.
    custom_removing_rule : int
        Percentage criteria of difference with previous RR-interval at which we consider
        that it is abnormal. If method is set to Karlsson, it is the percentage of difference
        between the absolute mean of previous and next RR-interval at which  to consider the beat
        as abnormal.
    verbose : bool
        Print information about ectopic beats.

    Returns
    ---------
    nn_intervals : list
        list of NN Interval
    outlier_count : int
        Count of outlier detected in RR-interval list

    References
    ----------
    .. [5] Kamath M.V., Fallen E.L.: Correction of the Heart Rate Variability Signal for Ectopics \
    and Miss- ing Beats, In: Malik M., Camm A.J.

    .. [6] Geometric Methods for Heart Rate Variability Assessment - Malik M et al
    """
    if method not in [MALIK_RULE, KAMATH_RULE, KARLSSON_RULE, ACAR_RULE, CUSTOM_RULE]:
        raise ValueError("Not a valid method. Please choose between malik, kamath, karlsson, acar.\
         You can also choose your own removing critera with custom_rule parameter.")

    if method == KARLSSON_RULE:
        nn_intervals, outlier_count = _remove_outlier_karlsson(rr_intervals=rr_intervals,
                                                               removing_rule=custom_removing_rule)

    elif method == ACAR_RULE:
        nn_intervals, outlier_count = _remove_outlier_acar(rr_intervals=rr_intervals)

    else:
        # set first element in list
        outlier_count = 0
        previous_outlier = False
        nn_intervals = [rr_intervals[0]]
        for i, rr_interval in enumerate(rr_intervals[:-1]):

            if previous_outlier:
                nn_intervals.append(rr_intervals[i + 1])
                previous_outlier = False
                continue

            if is_outlier(rr_interval, rr_intervals[i + 1], method=method,
                          custom_rule=custom_removing_rule):
                nn_intervals.append(rr_intervals[i + 1])
            else:
                nn_intervals.append(np.nan)
                outlier_count += 1
                previous_outlier = True

    if verbose:
        print("{} ectopic beat(s) have been deleted with {} rule.".format(outlier_count, method))

    return nn_intervals

def interpolate_nan_values(rr_intervals: list,
                           interpolation_method: str = "linear",
                           limit_area: str = None,
                           limit_direction: str = "forward",
                           limit=None,) -> list:
    """
    Function that interpolate Nan values with linear interpolation

    Parameters
    ---------
    rr_intervals : list
        RrIntervals list.
    interpolation_method : str
        Method used to interpolate Nan values of series.
    limit_area: str
        If limit is specified, consecutive NaNs will be filled with this restriction.
    limit_direction: str
        If limit is specified, consecutive NaNs will be filled in this direction.
    limit: int
        TODO
    Returns
    ---------
    interpolated_rr_intervals : list
        new list with outliers replaced by interpolated values.
    """
    # search first nan data and fill it post value until it is not nan
    if np.isnan(rr_intervals[0]):
        start_idx = 0

        while np.isnan(rr_intervals[start_idx]):
            start_idx += 1

        rr_intervals[0:start_idx] = [rr_intervals[start_idx]] * start_idx
    else:
        pass
    # change rr_intervals to pd series
    series_rr_intervals_cleaned = pd.Series(rr_intervals)
    # Interpolate nan values and convert pandas object to list of values
    interpolated_rr_intervals = series_rr_intervals_cleaned.interpolate(method=interpolation_method,
                                                                        limit=limit,
                                                                        limit_area=limit_area,
                                                                        limit_direction=limit_direction)
    return interpolated_rr_intervals.values.tolist()

def is_outlier(rr_interval: int, next_rr_interval: float, method: str = "malik",
               custom_rule: float = 0.2) -> bool:
    """
    Test if the rr_interval is an outlier

    Parameters
    ----------
    rr_interval : int
        RrInterval
    next_rr_interval : int
        consecutive RrInterval
    method : str
        method to use to clean outlier. malik, kamath, karlsson, acar or custom
    custom_rule : int
        percentage criteria of difference with previous RR-interval at which we consider
        that it is abnormal

    Returns
    ----------
    outlier : bool
        True if RrInterval is valid, False if not
    """
    if method == MALIK_RULE:
        outlier = abs(rr_interval - next_rr_interval) <= 0.2 * rr_interval
    elif method == KAMATH_RULE:
        outlier = 0 <= (next_rr_interval - rr_interval) <= 0.325 * rr_interval or 0 <= \
                  (rr_interval - next_rr_interval) <= 0.245 * rr_interval
    else:
        outlier = abs(rr_interval - next_rr_interval) <= custom_rule * rr_interval
    return outlier