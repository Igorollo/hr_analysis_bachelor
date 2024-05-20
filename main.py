import reading_file
import analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
import pingouin as pg
import rysowanie


df_osobno, df_razem, df_r_sur, df_o_sur, lista_imion, li = reading_file.wczytaj("pomiary_licencjat_ostat.csv")
osobno = np.concatenate(df_osobno) 
razem = np.concatenate(df_razem)
syg_all = np.concatenate((osobno, razem))
sur_ans_o = np.concatenate(df_o_sur)
sur_ans_o = [i for i in sur_ans_o[::2]]
sur_ans_r = np.concatenate(df_r_sur)
sur_ans_r = [i for i in sur_ans_r[::2]]

lista_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r = np.zeros((len(df_razem), df_razem[0].shape[0]//2))
lista_o_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_o_freq = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_freq = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_corr_hr_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_corr_hrv_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_corr_hr_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_corr_hrv_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_freq_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_freq_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
listeczka = []
for i in range(len(df_osobno)):
    for k, i_o in enumerate(range(0, df_osobno[0].shape[0]-1, 2)):
        df_osobno0 = analysis.standarize_data(df_osobno[i].iloc[i_o])
        df_osobno1 = analysis.standarize_data(df_osobno[i].iloc[i_o+1])
        df_o_intpol0 = analysis.interpolate_data(df_osobno0) 
        df_o_intpol1 = analysis.interpolate_data(df_osobno1)
        lista_o[i, k], czas_max_corr_hr_o[i,k] = analysis.rolling_corr(df_o_intpol0, df_o_intpol1)

        df_razem0 = analysis.standarize_data(df_razem[i].iloc[i_o])
        df_razem1 = analysis.standarize_data(df_razem[i].iloc[i_o+1])
        df_r_intpol0 = analysis.interpolate_data(df_razem0) 
        df_r_intpol1 = analysis.interpolate_data(df_razem1)
        lista_r[i, k], czas_max_corr_hr_r[i,k] = analysis.rolling_corr(df_r_intpol0, df_r_intpol1)
lista_org = lista_r
nazwa = "HR"
lista_imion1 = np.concatenate(li)
lista = np.concatenate(df_razem)
lista, lista_czas = analysis.compars(lista_imion1, lista)
plt.hist(lista_czas)
plt.savefig(nazwa+'czas.png', bbox_inches='tight')
plt.show()

plt.hist(lista, color = 'b')
    #print(len(lista_imion1))
    #print(len(lista))
analysis.normality_check(lista)
for warunki, strs, color in zip(lista_org, ["nieznajomi", "znajomi", "pary"], ['g', 'c', 'k']):
        #plt.hist(lista, density=True)
        plt.axvline(np.mean(warunki), label = strs, color = color)
        with open(nazwa+".txt", "a") as file1:
            # Writing data to a file
            file1.write("\n" + strs + "\n")
            file1.writelines(str(stats.ttest_ind(a=lista, b=warunki)))

lista_org = np.concatenate(lista_r)
plt.axvline(np.mean(lista_org), label = "razem", color = 'r')
with open(nazwa+".txt", "a") as file1:
            # Writing data to a file
        file1.write("\n" + "RAZEM "+ "\n")
        file1.writelines(str(stats.ttest_ind(a=lista, b= lista_org))) 
plt.legend()
plt.savefig(nazwa+'corr.png', bbox_inches='tight')
plt.show()
"""
for lista_o, lista_r, tytul, czas_max_corr_hr_o, czas_max_corr_hr_r in zip([lista_o],[lista_r],['HR'], [czas_max_corr_hr_o], [czas_max_corr_hr_r]):
    all_osobno = np.concatenate(lista_o)
    all_razem = np.concatenate(lista_r)
    all_ = np.concatenate((all_osobno, all_razem))
    czas_all_osobno = np.concatenate(czas_max_corr_hr_o)
    czas_all_razem = np.concatenate(czas_max_corr_hr_r)
    rysowanie.histogram_ro(czas_all_osobno, czas_all_razem)
    

    rysowanie.znajomosc_scatter(lista_o, lista_r, tytul)
    rysowanie.histogram_ro_znaj(lista_o, 'osobno ')
    rysowanie.histogram_ro_znaj(lista_r, 'razem ')
    rysowanie.histogram_ro(all_osobno, all_razem)
    rysowanie.histogram_roznic(lista_o, lista_r)
    rysowanie.znajomosc_scatter_roznica(lista_o, lista_r)
    rysowanie.scatter_ro(all_osobno, all_razem)
    rysowanie.scatter_urcs_znaj(sur_ans_o, all_osobno, sur_ans_r, all_razem)
    rysowanie.scatter_urcs_ro(sur_ans_r, all_razem, all_osobno)

    roznice = lista_r-lista_o

    print('\n'+tytul)
    
    print("KORELACJA")
    print(stats.pearsonr(sur_ans_r, all_razem, alternative='two-sided'))
    print(stats.pearsonr(sur_ans_o, all_osobno, alternative='two-sided'))
    print(stats.pearsonr(sur_ans_r, all_razem-all_osobno, alternative='two-sided'))

    print(f_oneway(lista_o[0], lista_o[1], lista_o[2]))

    print(f_oneway(lista_r[0], lista_r[1], lista_r[2]))

    print(f_oneway(roznice[0], roznice[1], roznice[2]))

    print(stats.ttest_rel(a=all_osobno, b=all_razem))
    print("RAZEM", all_razem)

    razem_relacja = np.concatenate((lista_r[1], lista_r[2]))
    print(stats.ttest_ind(a=razem_relacja, b=lista_r[0], alternative='greater'))

    relacja_roznice = np.concatenate((roznice[1], roznice[2]))
    print(stats.ttest_ind(a=relacja_roznice, b=roznice[0], alternative='greater'))


    print("sprawdzanie normalności danych")
    for check in range(0,3):
        print("razem" + str(check))
        analysis.normality_check(lista_r[check])
    for check in range(0,3):
        print("osobno" + str(check))
        analysis.normality_check(lista_o[check])
    print("normalnosc wszysktich danych")
    analysis.normality_check(all_)
    print("normalnosc wszysktich danych osobno")
    analysis.normality_check(all_osobno)
    print("normalnosc wszysktich danych razem")
    analysis.normality_check(all_razem)


    result = [0 for i in range(all_osobno.shape[0])] + [1 for i in range(all_razem.shape[0])]
    znajomosci = [0 for i in range(lista_o[0].shape[0])] + [1 for i in range(lista_o[1].shape[0])]+ [2 for i in range(lista_o[2].shape[0])]+ [0 for i in range(lista_r[0].shape[0])]+ [1 for i in range(lista_r[1].shape[0])]+ [2 for i in range(lista_r[2].shape[0])]

    subjects = list(range(1, len(all_)//2+1))+list(range(1, len(all_)//2+1))


    df = pd.DataFrame({
        "subid": subjects,
    "korelacje": all_,
    "or": result,
    "znajomosci": znajomosci })
    aov = pg.anova(data=df, dv="korelacje", between=["or", "znajomosci"], detailed=True)
    print(aov) """





    



