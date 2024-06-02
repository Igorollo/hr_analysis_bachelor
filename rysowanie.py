import matplotlib.pyplot as plt
import numpy as np
import analysis
color_razem = 'teal'
color_osono ='goldenrod'
def znajomosc_scatter(lista_o, lista_r, tytul):
    mean_lista_o = [np.mean(i) for i in lista_o]
    std_lista_o = [np.std(i) for i in lista_o]
    mean_lista_r = [np.mean(i) for i in lista_r]
    std_lista_r = [np.std(i) for i in lista_r]

    # Add the mean to the scatter plot

    plt.scatter(['osobno nieznajomi']*lista_o[0].shape[0], lista_o[0], s=10)
    plt.scatter(['razem nieznajomi']*lista_r[0].shape[0], lista_r[0], s=10)
    plt.scatter(['osobno znajomi']*lista_o[1].shape[0], lista_o[1], s=10)
    plt.scatter(['razem znajomi']*lista_r[1].shape[0], lista_r[1], s=10)
    plt.scatter(['osobno para']*lista_o[2].shape[0], lista_o[2], s=10)
    plt.scatter(['razem para']*lista_r[2].shape[0], lista_r[2], s=10)

    for i, label in enumerate(['osobno nieznajomi', 'osobno znajomi', 'osobno para']):
        if i == 2:
            plt.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3, label = 'mean with std')
            break
        plt.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3)
    for i, label in enumerate(['razem nieznajomi', 'razem znajomi', 'razem para']):
        plt.errorbar(label, mean_lista_r[i], yerr=std_lista_r[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3)

    x_ticks = ['osobno nieznajomi', 'razem nieznajomi', 'osobno znajomi', 'razem znajomi', 'osobno para', 'razem para']
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.title(tytul)
    plt.legend()
    plt.show()
def znajomosc_scatter_ax(lista_o, lista_r,tytul, ax):
    mean_lista_o = [np.mean(i) for i in lista_o]
    std_lista_o = [np.std(i) for i in lista_o]
    mean_lista_r = [np.mean(i) for i in lista_r]
    std_lista_r = [np.std(i) for i in lista_r]

    # Add the mean to the scatter plot

    ax.scatter(['osobno nieznajomi']*lista_o[0].shape[0], lista_o[0], s=10)
    ax.scatter(['razem nieznajomi']*lista_r[0].shape[0], lista_r[0], s=10)
    ax.scatter(['osobno znajomi']*lista_o[1].shape[0], lista_o[1], s=10)
    ax.scatter(['razem znajomi']*lista_r[1].shape[0], lista_r[1], s=10)
    ax.scatter(['osobno para']*lista_o[2].shape[0], lista_o[2], s=10)
    ax.scatter(['razem para']*lista_r[2].shape[0], lista_r[2], s=10)

    for i, label in enumerate(['osobno nieznajomi', 'osobno znajomi', 'osobno para']):
        if i == 2:
            ax.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3, label = 'mean with std')
            break
        ax.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3)
    for i, label in enumerate(['razem nieznajomi', 'razem znajomi', 'razem para']):
        ax.errorbar(label, mean_lista_r[i], yerr=std_lista_r[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3)

    x_ticks = ['osobno nieznajomi', 'razem nieznajomi', 'osobno znajomi', 'razem znajomi', 'osobno para', 'razem para']
    ax.set_xticks(range(len(x_ticks)), x_ticks)
    ax.set_title(tytul)
    ax.set_ylabel("korelacje")
    plt.legend()


def znajomosc_scatter_roznica(lista_o, lista_r):
    lista = lista_r-lista_o
    mean_lista_o = [np.mean(i) for i in lista]
    std_lista_o = [np.std(i) for i in lista]

    # Add the mean to the scatter plot

    plt.scatter(['nieznajomi']*lista[0].shape[0], lista[0], s=10)
    plt.scatter(['znajomi']*lista[1].shape[0], lista[1], s=10)
    plt.scatter(['para']*lista[2].shape[0], lista[2], s=10)

    for i, label in enumerate(['nieznajomi', 'znajomi', 'para']):
        if i == 2:
            plt.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3, label = 'mean with std')
            break
        plt.errorbar(label, mean_lista_o[i], yerr=std_lista_o[i], fmt='x', color='dimgray', ecolor='dimgray', elinewidth=1, capsize=3)

    x_ticks = ['nieznajomi','znajomi', 'para']
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.ylabel('roznica korelacji r-o')
    plt.legend()
    plt.show()

def histogram_roznic(lista_o, lista_r):
    lista = lista_r-lista_o

    plt.hist(lista[0], label='nieznajomi', bins = 14, alpha=0.5)
    plt.hist(lista[1], label='znajomi', bins = 14, alpha=0.5)
    plt.hist(lista[2], label='pary', bins = 14, alpha=0.5)
    plt.xlabel('roznica korelacji r-o')
    plt.legend()
    plt.show()

def histogram_ro_znaj(lista_r, label):
    for i, color, type in zip([0,1,2],['goldenrod', 'orchid', 'teal'], ['nieznajomi', 'znajomi', 'para']):
        plt.hist(lista_r[i], bins=14, color=color, alpha=0.7, label=label+str(type))
    plt.legend()
    plt.show()

def scatter_ro(all_osobno, all_razem):
    plt.scatter(['osobno']*all_osobno.shape[0], all_osobno, s=10, color=color_osono, label= 'osobno')
    plt.scatter(['razem']*all_razem.shape[0], all_razem, s=10, color = color_razem, label = 'razem')
    plt.errorbar(['osobno'], np.mean(all_osobno), yerr=np.std(all_osobno), fmt='x', color='black', ecolor='dimgray', elinewidth=1, capsize=3, label = 'mean with std')
    plt.errorbar(['razem'], np.mean(all_razem), yerr=np.std(all_razem), fmt='x', color='black', ecolor='dimgray', elinewidth=1, capsize=3)
    x_ticks = ['osobno', 'razem']
    plt.plot(x_ticks, [np.mean(all_osobno)]*2,color='goldenrod')
    plt.plot(x_ticks, [np.mean(all_razem)]*2,color='teal')
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.legend()
    plt.show()

def histogram_ro(all_osobno, all_razem):
    plt.rc('legend',fontsize='x-large') 
    plt.hist(all_osobno, bins=10, color='goldenrod', alpha=0.7, label='osobno')
    plt.hist(all_razem, bins=10, color='teal', alpha=0.7, label='razem')
    plt.legend()
    plt.show()
def histogram_ro_ax(all_osobno, all_razem,title, ax):
    ax.hist(all_osobno, bins=10, color='goldenrod', alpha=0.7, label='osobno')
    ax.hist(all_razem, bins=10, color='teal', alpha=0.7, label='razem')
    ax.set_xlabel("korelacje")
    ax.set_title(title)
    ax.legend()
    
def scatter_urcs_znaj(sur_ans_o, all_osobno, sur_ans_r, all_razem):
    plt.scatter(sur_ans_o, all_osobno, label = 'osobno', color = color_osono)
    plt.scatter(sur_ans_r, all_razem, label = 'razem',color= color_razem)
    plt.xlabel('URCS score')
    plt.legend()
    plt.show()

def scatter_urcs_ro(sur_ans_r, all_razem, all_osobno):
    plt.scatter(sur_ans_r, all_razem-all_osobno)
    plt.xlabel('URCS score')
    plt.ylabel('różnice korelacji razem-osobno')
    plt.show()

def minmax_plot(df_osobno, df_razem, all_razem, all_osobno, sek, tytul):
    fig, axs = plt.subplots(2, 2, figsize=(12, 4))
    max1 = analysis.interpolate_data(analysis.standarize_data(np.concatenate(df_razem)[all_razem.argmax()*2]))
    max2 = analysis.interpolate_data(analysis.standarize_data(np.concatenate(df_razem)[all_razem.argmax()*2+1]))
    min1 = analysis.interpolate_data(analysis.standarize_data(np.concatenate(df_osobno)[all_osobno.argmin()*2]))
    min2 = analysis.interpolate_data(analysis.standarize_data(np.concatenate(df_osobno)[all_osobno.argmin()*2+1]))
    minimum = min(len(max1), len(max2), len(min1), len(min2))
    max1 = max1[:minimum]
    max2 = max2[:minimum]
    min1 = min1[:minimum]
    min2 = min2[:minimum]
    czas_lista_rr = np.linspace(0,6*60,minimum)
    axs[0,1].plot(czas_lista_rr,max1)
    axs[0,1].plot(czas_lista_rr,max2)
    axs[0,1].set_xlim([260,320])
    axs[0,1].set_title("maksymalna korelacja")

    axs[0,0].plot(czas_lista_rr,min1)
    axs[0,0].plot(czas_lista_rr,min2)
    axs[0,0].set_xlim([260,320])
    axs[0,0].set_ylabel("interwały NN [ms]")
    axs[0,0].set_title("minimalna korelacja")
    overlap = int(sek/2)
    if tytul[:-4] == "rmssd":
        max1_hrv = analysis.rmssd_p(max1, overlap)
        max2_hrv = analysis.rmssd_p(max2, overlap)
        min1_hrv = analysis.rmssd_p(min1, overlap)
        min2_hrv = analysis.rmssd_p(min2, overlap)
    elif tytul[:-4] == "stdhr":
        max1_hrv = analysis.hrsd_p(max1, overlap)
        max2_hrv = analysis.hrsd_p(max2, overlap)
        min1_hrv = analysis.hrsd_p(min1, overlap)
        min2_hrv = analysis.hrsd_p(min2, overlap)
    elif tytul[:-4] == "stdnn":
        max1_hrv = analysis.nnsd_p(max1, overlap)
        max2_hrv = analysis.nnsd_p(max2, overlap)
        min1_hrv = analysis.nnsd_p(min1, overlap)
        min2_hrv = analysis.nnsd_p(min2, overlap)
    czas_lista_hrv = np.linspace(0,6*60, len(max1_hrv))
    axs[1,1].plot(czas_lista_hrv,max1_hrv)
    axs[1,1].plot(czas_lista_hrv,max2_hrv)
    axs[1,1].set_xlim([260,320])
    axs[1,1].set_xlabel("czas [s]")

    axs[1,0].plot(czas_lista_hrv,min1_hrv)
    axs[1,0].plot(czas_lista_hrv,min2_hrv)
    axs[1,0].set_xlim([260,320])
    axs[1,0].set_xlabel("czas [s]")
    axs[1,0].set_ylabel(tytul[:-4].upper()+" [ms]")
    plt.tight_layout()
    plt.show()


def scatter_urcs_znaj_ax(sur_ans_r, all_razem, tytul, ax):
    ax.scatter(sur_ans_r, all_razem, color= color_razem)
    ax.set_xlabel('URCS score')
    ax.set_ylabel("korelacje")
    ax.set_title(tytul)
    
   