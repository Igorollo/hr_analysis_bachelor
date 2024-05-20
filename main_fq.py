import reading_file
import analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
czas_max_corr_hrv_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_max_corr_hrv_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_nn_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_hrv_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_nn_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
czas_hrv_r = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))

lista_o_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_o_nnstd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_nnstd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_o_hrvstd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_hrvstd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
listeczka = []
for i in range(len(df_osobno)):
    for k, i_o in enumerate(range(0, df_osobno[0].shape[0]-1, 2)):
        df_osobno0 = analysis.standarize_data(df_osobno[i].iloc[i_o])
        df_osobno1 = analysis.standarize_data(df_osobno[i].iloc[i_o+1])
        df_o_intpol0 = analysis.interpolate_data(df_osobno0) 
        df_o_intpol1 = analysis.interpolate_data(df_osobno1)
        lista_o_rmssd[i,k], czas_max_corr_hrv_o[i,k] = analysis.rmssd_corr(df_o_intpol0, df_o_intpol1, 30)
        lista_o_hrvstd[i,k], czas_hrv_o[i,k] = analysis.stdhrv_corr(df_o_intpol0, df_o_intpol1, 30)
        lista_o_nnstd[i,k], czas_nn_o[i,k] = analysis.stdnn_corr(df_o_intpol0, df_o_intpol1, 30)

        df_razem0 = analysis.standarize_data(df_razem[i].iloc[i_o])
        df_razem1 = analysis.standarize_data(df_razem[i].iloc[i_o+1])
        df_r_intpol0 = analysis.interpolate_data(df_razem0) 
        df_r_intpol1 = analysis.interpolate_data(df_razem1)
        lista_r_rmssd[i,k], czas_max_corr_hrv_r[i,k] = analysis.rmssd_corr(df_r_intpol0, df_r_intpol1, 29)
        lista_r_hrvstd[i,k], czas_hrv_r[i,k] = analysis.stdhrv_corr(df_r_intpol0, df_r_intpol1, 29)
        lista_r_nnstd[i,k], czas_nn_r[i,k] = analysis.stdnn_corr(df_r_intpol0, df_r_intpol1, 29)
czas = "30"
for lista_org, nazwa in zip([lista_r_rmssd, lista_r_hrvstd, lista_r_nnstd], ["rmssd_por", "hrvstd_por", "nnstd_por"]):
    lista_imion1 = np.concatenate(li)
    lista = np.concatenate(df_razem)
    lista, lista_czas = analysis.compars_f(lista_imion1, lista, 29, nazwa)
    plt.hist(lista_czas)
    plt.savefig(nazwa+czas+'czas.png', bbox_inches='tight')
    plt.show()

    plt.hist(lista, color = 'b')
    #print(len(lista_imion1))
    #print(len(lista))
    analysis.normality_check(lista)
    for warunki, strs, color in zip(lista_org, ["nieznajomi", "znajomi", "pary"], ['g', 'c', 'k']):
        #plt.hist(lista, density=True)
        plt.axvline(np.mean(warunki), label = strs, color = color)
        with open(nazwa+czas+".txt", "a") as file1:
            # Writing data to a file
            file1.write("\n" + "\n" + strs + "\n")
            file1.writelines(str(stats.ttest_ind(a=lista, b=warunki)))

    lista_org = np.concatenate(lista_r_rmssd)
    plt.axvline(np.mean(lista_org), label = "razem", color = 'r')
    with open(nazwa+czas+".txt", "a") as file1:
            # Writing data to a file
        file1.write("\n" + "RAZEM "+ "\n")
        file1.writelines(str(stats.ttest_ind(a=lista, b= lista_org))) 
    plt.legend()
    plt.savefig(nazwa+'corr.png', bbox_inches='tight')
    plt.show()



"""
for lista_o, lista_r, tytul, czas_max_corr_hrv_o, czas_max_corr_hrv_r in zip([lista_o_rmssd, lista_o_hrvstd ,lista_o_nnstd],[lista_r_rmssd, lista_r_hrvstd, lista_r_nnstd],['rmssd.txt', 'stdhrv.txt', "stdnn.txt"], [czas_max_corr_hrv_o, czas_hrv_o, czas_nn_o], [czas_max_corr_hrv_r, czas_hrv_r, czas_nn_r]):
    all_osobno = np.concatenate(lista_o)
    all_razem = np.concatenate(lista_r)
    all_ = np.concatenate((all_osobno, all_razem))
    
    rysowanie.histogram_ro(np.concatenate(czas_max_corr_hrv_o), np.concatenate(czas_max_corr_hrv_r))
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
    all_osobno1 = [i for i in all_osobno if i < 3]
    analysis.normality_check(lista_o[1])
    print("normalnosc wszysktich danych razem")
    analysis.normality_check(all_razem)
    #plt.hist(lista_r[0])
    #plt.show()

    #plt.hist(lista_r[1])
    #plt.show()


    result = [0 for i in range(all_osobno.shape[0])] + [1 for i in range(all_razem.shape[0])]
    znajomosci = [0 for i in range(lista_o[0].shape[0])] + [1 for i in range(lista_o[1].shape[0])]+ [2 for i in range(lista_o[2].shape[0])]+ [0 for i in range(lista_r[0].shape[0])]+ [1 for i in range(lista_r[1].shape[0])]+ [2 for i in range(lista_r[2].shape[0])]

    subjects = list(range(1, len(all_)//2+1))+list(range(1, len(all_)//2+1))


    df = pd.DataFrame({
        "subid": subjects,
    "korelacje": all_,
    "or": result,
    "znajomosci": znajomosci })
    aov = pg.mixed_anova(dv='korelacje', between='znajomosci',
                  within='or', subject='subid', data=df)
    razem_relacja = np.concatenate((lista_r[1], lista_r[2]))

    pg.plot_paired(data=df, dv='korelacje', within='or', subject='subid')
    plt.show()
    with open(tytul, "a") as file1:
        # Writing data to a file
        file1.writelines("\n" + "dla 15 sekundowego okienka" +"\n" )

        file1.writelines("r pearsona dla danych osobno " + str(stats.pearsonr(sur_ans_o, all_osobno, alternative='two-sided')[1])+"\n")
        file1.writelines("r pearsona dla danych razem "+ str(stats.pearsonr(sur_ans_r, all_razem, alternative='two-sided')[1])+"\n")
        file1.writelines("r pearsona dla danych razem-osobno "+ str(stats.pearsonr(sur_ans_r, all_razem-all_osobno, alternative='two-sided')[1])+"\n")
        file1.writelines("mixed_anova " + str(aov)+"\n")"""







