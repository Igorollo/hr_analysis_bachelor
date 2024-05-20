import matplotlib.pyplot as plt
import numpy as np

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
    plt.scatter(['osobno']*all_osobno.shape[0], all_osobno, s=10, color='goldenrod')
    plt.scatter(['razem']*all_razem.shape[0], all_razem, s=10, color = 'teal')
    plt.errorbar(['osobno'], np.mean(all_osobno), yerr=np.std(all_osobno), fmt='x', color='black', ecolor='dimgray', elinewidth=1, capsize=3, label = 'mean with std')
    plt.errorbar(['razem'], np.mean(all_razem), yerr=np.std(all_razem), fmt='x', color='black', ecolor='dimgray', elinewidth=1, capsize=3)
    x_ticks = ['osobno', 'razem']
    plt.plot(x_ticks, [np.mean(all_osobno)]*2,color='goldenrod')
    plt.plot(x_ticks, [np.mean(all_razem)]*2,color='teal')
    plt.xticks(range(len(x_ticks)), x_ticks)
    plt.legend()
    plt.show()

def histogram_ro(all_osobno, all_razem):
    plt.hist(all_osobno, bins=10, color='goldenrod', alpha=0.7, label='osobno')
    plt.hist(all_razem, bins=10, color='teal', alpha=0.7, label='razem')
    plt.legend()
    plt.show()

def scatter_urcs_znaj(sur_ans_o, all_osobno, sur_ans_r, all_razem):
    plt.scatter(sur_ans_o, all_osobno, label = 'osobno')
    plt.scatter(sur_ans_r, all_razem, label = 'razem')
    plt.xlabel('URCS score')
    plt.legend()
    plt.show()

def scatter_urcs_ro(sur_ans_r, all_razem, all_osobno):
    plt.scatter(sur_ans_r, all_razem-all_osobno)
    plt.xlabel('URCS score')
    plt.ylabel('różnice korelacji razem-osobno')
    plt.show()