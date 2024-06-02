import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from hrvanalysis_ import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from scipy.interpolate import CubicSpline
from scipy import stats
import pandas as pd
from scipy import signal
import pingouin as pg
def wykreslanka(razem, osobno, tytul):
    for i in range(0,len(razem)):
        all_razem = np.concatenate((razem[:i],razem[i+1:]))
        all_osobno = np.concatenate((osobno[:i],osobno[i+1:]))
        all_ = np.concatenate((all_razem, all_osobno))
        result = [0 for i in range(all_osobno.shape[0])] + [1 for i in range(all_razem.shape[0])]
        if i<12: znajomosci = [0 for i in range(11)] + [1 for i in range(12)]+ [2 for i in range(12)]+ [0 for i in range(11)]+ [1 for i in range(12)]+ [2 for i in range(12)]
        elif 12<=i<24: znajomosci = [0 for i in range(12)] + [1 for i in range(11)]+ [2 for i in range(12)]+ [0 for i in range(12)]+ [1 for i in range(11)]+ [2 for i in range(12)]
        elif 24<=i<36: znajomosci = [0 for i in range(12)] + [1 for i in range(12)]+ [2 for i in range(11)]+ [0 for i in range(12)]+ [1 for i in range(12)]+ [2 for i in range(11)]
        else: print("ERRRRRORRRRRRRRRR")
        subjects = list(range(1, len(all_)//2+1))+list(range(1, len(all_)//2+1))


        df = pd.DataFrame({
            "subid": subjects,
        "korelacje": all_,
        "or": result,
        "znajomosci": znajomosci })
        aov = pg.mixed_anova(dv='korelacje', between='znajomosci',
                    within='or', subject='subid', data=df)
        with open("wykrelanie"+tytul, "a") as file1:
                # Writing data to a file
                file1.writelines("\n")
                file1.writelines(str(aov))

def compars(names, data):
    lista = []
    lista_czas = []
    data = [standarize_data(i) for i in data]
    data = [interpolate_data(i) for i in data]
    for i in range(len(names)):
        for j in range(i, len(names)):
            if names[i] != names[j]:
                if i % 2:
                    if j != i-1:
                            p, t =rolling_corr(data[i], data[j])
                            lista.append(p)
                            lista_czas.append(t)
                       
                if i % 2==0:
                    if j != i+1:
                            p, t =rolling_corr(data[i], data[j])
                            lista.append(p)
                            lista_czas.append(t)

    return lista, lista_czas

def compars_f(names, data, sek, nazwa, pre = True):
    lista = []
    lista_czas = []
    if pre:
        data = [standarize_data(i) for i in data]
        data = [interpolate_data(i) for i in data]
    for i in range(len(names)):
        for j in range(i, len(names)):
            if names[i] != names[j]:
                if i % 2:
                    if j != i-1:
                        if nazwa == "rmssd_por":
                            p, t =rmssd_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)
                        elif nazwa == "hrvstd_por":
                            p, t =stdhrv_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)
                        elif nazwa == "nnstd_por":
                            p, t =stdnn_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)
                elif i % 2==0:
                    if j != i+1:
                        if nazwa == "rmssd_por":
                            p, t =rmssd_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)
                        elif nazwa == "hrvstd_por":
                            p, t =stdhrv_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)
                        elif nazwa == "nnstd_por":
                            p, t =stdnn_corr(data[i], data[j], sek)
                            lista.append(p)
                            lista_czas.append(t)

    return lista, lista_czas
        
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
    """plt.scatter(x_ecg, rr_ecg, color = "orange", label = 'Orginalny sygnał')
    plt.plot(xx_ecg[40:], rr_interpolated_ecg[40:], label = "Sygnał po interpolacji")
    plt.xlabel("Czas (s)")
    plt.ylabel("Odstępy czasowe pomiędzy kolejnymi załamkami N (ms)")
    plt.xlim(0,60)
    plt.legend()
    plt.show()"""
    return rr_interpolated_ecg[fs*5:]


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
    sw_stat, p_value2 = stats.shapiro(data1)
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(data1_standardized, stats.norm.cdf)
    print(f"Normality check SW = {sw_stat:.4f}, p-value = {p_value2:.4f}")
    print(f"Normality check KS = {ks_statistic:.4f}, p-value = {p_value:.4f}")
    

def rolling_corr(data1, data2):

    minimum = min(len(data1), len(data2))
    data1 = pd.Series(1/data1[:minimum]) #TODO czy tak jest okey z minimum
    data2 = pd.Series(1/data2[:minimum])
    lista = []
    rang = [32-i for i in range(1,32)]
    for i in rang:
        lista.append(stats.pearsonr(data1[i:], data2[:-i],alternative='two-sided')[0])

    lista.append(stats.pearsonr(data1, data2, alternative='two-sided')[0])
    for i in range(1, 33):
        lista.append(stats.pearsonr(data2[i:], data1[:-i],alternative='two-sided')[0])
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-32)/4

def rmssd_corr(data1, data2, sek = 30, fs= 4):
    przes = fs*4
    overlap = int(fs*sek/2 - 1)
    minimum = min(len(data1), len(data2))
    data1 = pd.Series(data1[:minimum])
    data2 = pd.Series(data2[:minimum])
    czasek = []
    lista = []
    rang = [przes-i for i in range(0,przes)]
    
    
    odcinki_2 = []
    for k,j in enumerate(data2):
        if k%overlap == 0 and len(data2)-(k+1)>overlap and k != 0:
            odcinki_2.append(data2[(k-overlap):(k+overlap)])
    for i in rang:
        odcinki_1 = []
        for k,j in enumerate(data1[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data1[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data1[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(rssmd(odcinki_1[:minimum], odcinki_2[:minimum])) #TODO czy minimum tak jak przy 1/nny

    odcinki_2 = []
    for k,j in enumerate(data1):
        if k%overlap == 0 and len(data1)-(k+1)>overlap and k != 0:
            odcinki_2.append(data1[(k-overlap):(k+overlap)])
    for i in range(1, przes+1):
        odcinki_1 = []
        for k,j in enumerate(data2[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data2[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data2[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(rssmd(odcinki_1[:minimum], odcinki_2[:minimum])) #TODO czy minimum tak jak przy 1/nny
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-przes)/4

def rssmd(odcinki_1, odcinki_2):
    diff_nni_1 = [np.diff(odcinek) for odcinek in odcinki_1]
    rmssd_1 = [np.sqrt(np.sum(odcinek ** 2)/(len(odcinek)-1)) for odcinek in diff_nni_1]
    diff_nni_2 = [np.diff(odcinek) for odcinek in odcinki_2]
    rmssd_2 = [np.sqrt(np.sum(odcinek ** 2)/(len(odcinek)-1)) for odcinek in diff_nni_2]
    minimum = min(len(rmssd_1), len(rmssd_2))
    return stats.pearsonr(rmssd_1[:minimum], rmssd_2[:minimum], alternative='two-sided')[0]
def sdnn(odcinki_1, odcinki_2):
    std_1 = [np.std(odcinek) for odcinek in odcinki_1]
    std_2 = [np.std(odcinek) for odcinek in odcinki_2]
    minimum = min(len(std_1), len(std_2))
    return stats.pearsonr(std_1[:minimum], std_2[:minimum], alternative='two-sided')[0]
def stdhrv_corr(data1, data2, sek = 30, fs=4):
    przes = fs*4
    overlap = int(fs*sek/2 - 1)
    minimum = min(len(data1), len(data2))
    data1 = pd.Series(1/data1[:minimum])
    data2 = pd.Series(1/data2[:minimum])
    lista = []
    odcinki_2 = []
    rang = [przes-i for i in range(0,przes)]
    for k,j in enumerate(data1):
        if k%overlap == 0 and len(data1)-(k+1)>overlap and k != 0:
            odcinki_2.append(data1[(k-overlap):(k+overlap)])
    for i in range(1, przes+1):
        odcinki_1 = []
        for k,j in enumerate(data2[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data2[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data2[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(sdnn(odcinki_1[:minimum], odcinki_2[:minimum])) 
    
    odcinki_2 = []
    for k,j in enumerate(data2): #TODO odwróć kolejnosc
        if k%overlap == 0 and len(data1)-(k+1)>overlap and k != 0:
            odcinki_2.append(data1[(k-overlap):(k+overlap)])
    odcinki_1 = []
    for k,j in enumerate(data1):
        if k%overlap == 0 and len(data2)-(k+1)>overlap and k != 0:
            odcinki_1.append(data2[(k-overlap):(k+overlap)])
    lista.append(sdnn(odcinki_1, odcinki_2))
    odcinki_2 = []
    for k,j in enumerate(data2):
        if k%overlap == 0 and len(data2)-(k+1)>overlap and k != 0:
            odcinki_2.append(data2[(k-overlap):(k+overlap)])
    for i in rang:
        odcinki_1 = []
        for k,j in enumerate(data1[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data1[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data1[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(sdnn(odcinki_1[:minimum], odcinki_2[:minimum])) #TODO czy minimum tak jak przy 1/nny
    
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-przes)/4
def stdnn_corr(data1, data2, sek = 30, fs=4):
    przes = fs*4
    overlap = int(fs*sek/2 - 1)
    minimum = min(len(data1), len(data2))
    data1 = pd.Series(data1[:minimum])
    data2 = pd.Series(data2[:minimum])
    lista = []
    rang = [przes-i for i in range(0,przes)]
    odcinki_2 = []
    for k,j in enumerate(data2):
        if k%overlap == 0 and len(data2)-(k+1)>overlap and k != 0:
            odcinki_2.append(data2[(k-overlap):(k+overlap)])
    for i in rang:
        odcinki_1 = []
        for k,j in enumerate(data1[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data1[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data1[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(sdnn(odcinki_1[:minimum], odcinki_2[:minimum])) #TODO czy minimum tak jak przy 1/nny
    
    odcinki_2 = []
    for k,j in enumerate(data2): #TODO odwróć kolejnosc
        if k%overlap == 0 and len(data1)-(k+1)>overlap and k != 0:
            odcinki_2.append(data1[(k-overlap):(k+overlap)])
    odcinki_1 = []
    for k,j in enumerate(data1):
        if k%overlap == 0 and len(data2)-(k+1)>overlap and k != 0:
            odcinki_1.append(data2[(k-overlap):(k+overlap)])
    lista.append(sdnn(odcinki_1, odcinki_2))
    
    odcinki_2 = []
    for k,j in enumerate(data1):
        if k%overlap == 0 and len(data1)-(k+1)>overlap and k != 0:
            odcinki_2.append(data1[(k-overlap):(k+overlap)])
    for i in range(1, przes+1):
        odcinki_1 = []
        for k,j in enumerate(data2[i:]): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(data2[i:])-(k+1)>overlap and k != 0: 
                odcinki_1.append(data2[(k-overlap+i):(k+overlap+i)])
        minimum = min(len(odcinki_1), len(odcinki_2))
        lista.append(sdnn(odcinki_1[:minimum], odcinki_2[:minimum])) #TODO czy minimum tak jak przy 1/nny
    
    return np.arctanh(np.max(lista)), (lista.index(np.max(lista))-przes)/4

def frequency_domain(nns_interpolated, fs=4):
    # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=nns_interpolated, fs=fs, nperseg=120)
    """ plt.plot(fxx, pxx)
    plt.xlim(0, 0.5)
    plt.xlabel("Częstości (Hz)")
    plt.ylabel("Amplituda")
    plt.show()"""
    
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
    return fxx, pxx
def rolling_corr_f(data1, data2, verbose = False):
    minimum = min(len(data1), len(data2))
    data1 = pd.Series(data1[:minimum])
    data2 = pd.Series(data2[:minimum])
    f1, p1 = frequency_domain(data1)
    f2, p2 = frequency_domain(data2)
    corr = stats.pearsonr(p1, p2, alternative='two-sided')[0]
    print(len(f1))
    if verbose and corr < 0.7:
        plt.plot(f1, p1)
        plt.plot(f2, p2)
        plt.show()
    return np.arctanh(corr)

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


def rmssd_p(syg_vis, overlap):
        odcinki_2 = []
        for k,j in enumerate(syg_vis): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(syg_vis)-(k+1)>overlap and k != 0:
                odcinki_2.append(syg_vis[(k-overlap):(k+overlap)])
        diff_nni_1 = [np.diff(odcinek) for odcinek in odcinki_2]
        rmssd_1 = [np.sqrt(np.sum(odcinek ** 2)/(len(odcinek)-1)) for odcinek in diff_nni_1]
        return rmssd_1
def nnsd_p(syg_vis, overlap):
    odcinki_2 = []
    for k,j in enumerate(syg_vis): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(syg_vis)-(k+1)>overlap and k != 0:
                odcinki_2.append(syg_vis[(k-overlap):(k+overlap)])
    std_1 = [np.std(odcinek) for odcinek in odcinki_2]
    return std_1
def hrsd_p(syg_vis, overlap):
    syg_vis = 1/syg_vis
    odcinki_2 = []
    for k,j in enumerate(syg_vis): #TODO odwróć kolejnosc
            if k%overlap == 0 and len(syg_vis)-(k+1)>overlap and k != 0:
                odcinki_2.append(syg_vis[(k-overlap):(k+overlap)])
    std_1 = [np.std(odcinek) for odcinek in odcinki_2]
    return std_1

def randomHR(signal):
    macierz = np.zeros((72,len(signal)))

    for k in range(macierz.shape[0]):
        random_values = np.random.normal(loc=800, scale=250, size=len(signal))
        macierz[k] = random_values
    return macierz