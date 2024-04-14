import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from hrvanalysis_ import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from scipy.interpolate import CubicSpline
from scipy import stats
import pandas as pd

def standarize_data(data):
    # rr_intervals_list contains integer values of RR-interval
    rr_intervals_list = data

    # This remove outliers from signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,  
                                                    low_rri=300, high_rri=1400, verbose=False)
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                    interpolation_method="linear")
    lenght = np.arange(0,len(interpolated_rr_intervals), 1)
    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
    return interpolated_nn_intervals

def interpolate_data(rr_ecg):
    x_ecg = np.cumsum(rr_ecg)/1000

    cs = CubicSpline(x_ecg, rr_ecg, extrapolate=True)
    # sample rate for interpolation
    fs = 4
    steps = 1 / fs

    # sample using the interpolation function
    xx_ecg = np.arange(0, np.max(x_ecg), steps)
    rr_interpolated_ecg = cs(xx_ecg)
    return rr_interpolated_ecg[20:]


def normality_check(data1, verbose = False):
    data1_standardized = stats.zscore(data1)
    # Create histograms
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.hist(data1_standardized, bins=30, color='blue', alpha=0.7, label='Data 1')
        plt.title('Histogram of Data 1')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(data1_standardized, stats.norm.cdf)
    print(f"Kolmogorov-Smirnov test for Data 1: KS statistic = {ks_statistic:.4f}, p-value = {p_value:.4f}")

def rolling_corr(data1, data2):

    minimum = min(len(data1), len(data2))
    data1 = pd.Series(1/data1[:minimum]) #TODO czy tak jest okey z minimum
    data2 = pd.Series(1/data2[:minimum])
    """data1 = (data1 - np.mean(data1)) / np.std(data1)
    data2 = (data2 - np.mean(data2)) / np.std(data2)"""
    lista = [stats.pearsonr(data1, data2, alternative='two-sided')[0]]
    for i in range(1, 32):
        lista.append(stats.pearsonr(data1[i:], data2[:-i],alternative='two-sided')[0])
    for i in range(1, 32):
        lista.append(stats.pearsonr(data2[i:], data1[:-i],alternative='two-sided')[0])
    return np.arctanh(np.max(lista))

def rmssd_corr(data1, data2, sek = 30):
    data1 = pd.Series(data1)
    data2 = pd.Series(data2)
    lista = []
    for i in range(1, 32): #odwroc kolejnosc
        odcinki_1 = []
        odcinek = []
        for k,j in enumerate(data1[i:]): #TODO odwróć kolejnosc
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_1.append(odcinek)
                odcinek = []
        odcinki_2 = []
        odcinek = []
        for k,j in enumerate(data2[:-i]):
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_2.append(odcinek)
                odcinek = []
        diff_nni_1 = [np.diff(odcinek) for odcinek in odcinki_1]
        rmssd_1 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_1]
        diff_nni_2 = [np.diff(odcinek) for odcinek in odcinki_2]
        rmssd_2 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_2]
        
        minimum = min(len(rmssd_1), len(rmssd_2))
        lista.append(stats.pearsonr(rmssd_1[:minimum], rmssd_2[:minimum], alternative='two-sided')[0]) #TODO czy minimum tak jak przy 1/nny
    for i in range(1, 32):
        odcinki_1 = []
        odcinek = []
        for k,j in enumerate(data2[i:]): #TODO odwróć kolejnosc
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_1.append(odcinek)
                odcinek = []
        odcinki_2 = []
        odcinek = []
        for k,j in enumerate(data1[:-i]):
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_2.append(odcinek)
                odcinek = []
        diff_nni_1 = [np.diff(odcinek) for odcinek in odcinki_1]
        rmssd_1 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_1]
        diff_nni_2 = [np.diff(odcinek) for odcinek in odcinki_2]
        rmssd_2 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_2]
        
        minimum = min(len(rmssd_1), len(rmssd_2))
        lista.append(stats.pearsonr(rmssd_1[:minimum], rmssd_2[:minimum], alternative='two-sided')[0]) #TODO czy minimum tak jak przy 1/nny
    return np.arctanh(np.max(lista))


    