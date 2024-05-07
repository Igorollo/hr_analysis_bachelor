import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from hrvanalysis_ import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from scipy.interpolate import CubicSpline
from scipy import stats
import pandas as pd
from scipy import signal
from scipy.integrate import trapz

def compars(names, data):
    lista = []
    data = [standarize_data(i) for i in data]
    data = [interpolate_data(i) for i in data]
    suma = 0
    for i in range(0,len(names),2):
        lista.append(rolling_corr(data[i], data[i+1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i+1 != j:
                lista.append(rolling_corr(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    for i in range(1,len(names),2):
        lista.append(rolling_corr(data[i], data[i-1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i-1 != j:
                lista.append(rolling_corr(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    with open("porównania.txt", "a") as file1:
        # Writing data to a file
        file1.write("\n")
        file1.writelines("ostateczna liczba osób ze statystycznie różna korelacją "+str(suma))
    return suma

def compars_f(names, data):
    lista = []
    data = [standarize_data(i) for i in data]
    data = [interpolate_data(i) for i in data]
    suma = 0
    for i in range(0,len(names),2):
        lista.append(rolling_corr_f(data[i], data[i+1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i+1 != j:
                lista.append(rolling_corr_f(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania_f.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    for i in range(1,len(names),2):
        lista.append(rolling_corr_f(data[i], data[i-1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i-1 != j:
                lista.append(rolling_corr_f(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania_f.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    with open("porównania_f.txt", "a") as file1:
        # Writing data to a file
        file1.write("\n")
        file1.writelines("ostateczna liczba osób ze statystycznie różna korelacją "+str(suma))
    return suma

def compars_rm(names, data):
    lista = []
    data = [standarize_data(i) for i in data]
    data = [interpolate_data(i) for i in data]
    suma = 0
    for i in range(0,len(names),2):
        lista.append(rmssd_corr(data[i], data[i+1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i+1 != j:
                lista.append(rmssd_corr(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania_rm.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    for i in range(1,len(names),2):
        lista.append(rmssd_corr(data[i], data[i-1])[0])
        for j in range(len(names)):
            if names[i] != names[j] and i != j and i-1 != j:
                lista.append(rmssd_corr(data[i], data[j])[0])
        normality_check(lista)
        z_scores = stats.zscore(lista)
        if  z_scores[0]>1.96:
            with open("porównania_rm.txt", "a") as file1:
                # Writing data to a file
                file1.write("\n")
                file1.writelines(names[i]+" "+str(z_scores[0]))
            suma += 1
        lista = []
    with open("porównania_rm.txt", "a") as file1:
        # Writing data to a file
        file1.write("\n")
        file1.writelines("ostateczna liczba osób ze statystycznie różna korelacją "+str(suma))
    return suma


        
def standarize_data(data):
    # rr_intervals_list contains integer values of RR-interval
    rr_intervals_list = data
    rr_intervals_list = [rri for rri in rr_intervals_list if not np.isnan(rri)]
    # This remove outliers from signal
    outlier_low = np.mean(rr_intervals_list) - 3 * np.std(rr_intervals_list)
    outlier_high = np.mean(rr_intervals_list) + 3 * np.std(rr_intervals_list)
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,  
                                                    low_rri=outlier_low, high_rri=outlier_high, verbose=False)
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                    interpolation_method="linear")
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
    print(f"Normality check KS = {ks_statistic:.4f}, p-value = {p_value:.4f}")

def rolling_corr(data1, data2):

    minimum = min(len(data1), len(data2))
    data1 = pd.Series(1/data1[:minimum]) #TODO czy tak jest okey z minimum
    data2 = pd.Series(1/data2[:minimum])
    lista = []
    for i in range(1, 32):
        lista.append(stats.pearsonr(data2[i:], data1[:-i],alternative='two-sided')[0])

    lista.append(stats.pearsonr(data1, data2, alternative='two-sided')[0])
    for i in range(1, 32):
        lista.append(stats.pearsonr(data1[i:], data2[:-i],alternative='two-sided')[0])
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-32)/4

def rmssd_corr(data1, data2, sek = 30):
    data1 = pd.Series(data1)
    data2 = pd.Series(data2)
    lista = []
    for i in range(1, 16):
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
        lista.append(rssmd(odcinki_1, odcinki_2)) #TODO czy minimum tak jak przy 1/nny
    
    odcinki_1 = []
    odcinek = []
    for k,j in enumerate(data2): #TODO odwróć kolejnosc
        odcinek.append(j)
        if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_1.append(odcinek)
                odcinek = []
    odcinki_2 = []
    odcinek = []
    for k,j in enumerate(data1):
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_2.append(odcinek)
                odcinek = []
    lista.append(rssmd(odcinki_1, odcinki_2))
    
    for i in range(1, 16): #odwroc kolejnosc
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
        lista.append(rssmd(odcinki_1, odcinki_2)) #TODO czy minimum tak jak przy 1/nny
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-16)/4

def rssmd(odcinki_1, odcinki_2):
    diff_nni_1 = [np.diff(odcinek) for odcinek in odcinki_1]
    rmssd_1 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_1]
    diff_nni_2 = [np.diff(odcinek) for odcinek in odcinki_2]
    rmssd_2 = [np.sqrt(np.mean(odcinek ** 2)) for odcinek in diff_nni_2]
    minimum = min(len(rmssd_1), len(rmssd_2))
    return stats.pearsonr(rmssd_1[:minimum], rmssd_2[:minimum], alternative='two-sided')[0]


def frequency_domain(nns_interpolated, fs=4):
    # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=nns_interpolated, fs=fs)
    
    '''
    Segement found frequencies in the bands 
     - Very Low Frequency (VLF): 0-0.04Hz 
     - Low Frequency (LF): 0.04-0.15Hz 
     - High Frequency (HF): 0.15-0.4Hz
    '''
    """cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)
    
    # calculate power in each band by integrating the spectral density 
    vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = trapz(pxx[cond_lf], fxx[cond_lf])
    hf = trapz(pxx[cond_hf], fxx[cond_hf])
    
    # sum these up to get total power
    total_power = vlf + lf + hf

    # find which frequency has the most power in each band
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

    # fraction of lf and hf
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)
    
    results = {}
    results['Power VLF (ms2)'] = vlf
    results['Power LF (ms2)'] = lf
    results['Power HF (ms2)'] = hf   
    results['Power Total (ms2)'] = total_power

    results['LF/HF'] = (lf/hf)
    results['Fraction LF (nu)'] = lf_nu
    results['Fraction HF (nu)'] = hf_nu"""
    return pxx
def rolling_corr_f(data1, data2):

    minimum = min(len(data1), len(data2))
    data1 = pd.Series(data1[:minimum])
    data2 = pd.Series(data2[:minimum])
    lista = []
    for i in range(1, 16):
        lista.append(stats.pearsonr(frequency_domain(data2[i:]), frequency_domain(data1[:-i]),alternative='two-sided')[0])

    lista.append(stats.pearsonr(frequency_domain(data1), frequency_domain(data2), alternative='two-sided')[0])
    for i in range(1, 16):
        lista.append(stats.pearsonr(frequency_domain(data1[i:]), frequency_domain(data2[:-i]),alternative='two-sided')[0])
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-16)/4

def freq_corr(data1, data2, sek = 30): #TODO if necesary
    data1 = pd.Series(data1)
    data2 = pd.Series(data2)
    lista = []
    results = {}
    results['Power VLF (ms2)'] = []
    results['Power LF (ms2)'] = []
    results['Power HF (ms2)'] = [] 
    results['Power Total (ms2)'] = []
    results['LF/HF'] = []
    results['Fraction LF (nu)'] = []
    results['Fraction HF (nu)'] = []
    results2 = results
    korelacje = results
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
        minimum = min(len(odcinki_1), len(odcinki_2))
        for i in range(minimum):
            freq1,_, p = frequency_domain(odcinki_1[i])
            freq2,_,p = frequency_domain(odcinki_2[i])
            for col, val in freq1.items():
                results[col].append(val)
            for col, val in freq2.items():
                results2[col].append(val)
        for key in results.keys():
            korelacje[key].append(stats.pearsonr(results[key], results2[key], alternative='two-sided')[0])

    odcinki_1 = []
    odcinek = []
    for k,j in enumerate(data2): #TODO odwróć kolejnosc
        odcinek.append(j)
        if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_1.append(odcinek)
                odcinek = []
    odcinki_2 = []
    odcinek = []
    for k,j in enumerate(data1):
            odcinek.append(j)
            if (k+1)%(sek*4-1) == 0: #30 sekund (fs=4)
                odcinki_2.append(odcinek)
                odcinek = []
    lista.append(rssmd(odcinki_1, odcinki_2))
    
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
        lista.append(rssmd(odcinki_1, odcinki_2)) #TODO czy minimum tak jak przy 1/nny
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-32)/4
    