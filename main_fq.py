import reading_file
import analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import rysowanie

sek = 30
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
        lista_o_rmssd[i,k], czas_max_corr_hrv_o[i,k] = analysis.rmssd_corr(df_o_intpol0, df_o_intpol1, sek)
        lista_o_hrvstd[i,k], czas_hrv_o[i,k] = analysis.stdhrv_corr(df_o_intpol0, df_o_intpol1, sek)
        lista_o_nnstd[i,k], czas_nn_o[i,k] = analysis.stdnn_corr(df_o_intpol0, df_o_intpol1, sek)

        df_razem0 = analysis.standarize_data(df_razem[i].iloc[i_o])
        df_razem1 = analysis.standarize_data(df_razem[i].iloc[i_o+1])
        df_r_intpol0 = analysis.interpolate_data(df_razem0) 
        df_r_intpol1 = analysis.interpolate_data(df_razem1)
        lista_r_rmssd[i,k], czas_max_corr_hrv_r[i,k] = analysis.rmssd_corr(df_r_intpol0, df_r_intpol1, sek)
        lista_r_hrvstd[i,k], czas_hrv_r[i,k] = analysis.stdhrv_corr(df_r_intpol0, df_r_intpol1, sek)
        lista_r_nnstd[i,k], czas_nn_r[i,k] = analysis.stdnn_corr(df_r_intpol0, df_r_intpol1, sek)
czas = str(sek+1)
"""lista_imion1 = np.concatenate(li)
lista = np.concatenate(df_razem)
lista_nnstd_r, lista_czas_nnstd_r = analysis.compars_f(lista_imion1, lista, sek, "nnstd_por")
lista = np.concatenate(df_osobno)
lista_nnstd_o, lista_czas_nnstd_o = analysis.compars_f(lista_imion1, lista, sek, "nnstd_por")
fig2, axs2 = plt.subplots(2, sharex=True, sharey=True)
#fig1, axs1 = plt.subplots(1, 2, sharex=True, sharey=True)
for lista_org, nazwa, axs, lista in zip([lista_o_nnstd, lista_r_nnstd], ["nnstd_por_o", "nnstd_por_r"], [axs2[0], axs2[1]], [lista_nnstd_o, lista_nnstd_r]):
    if "_o" in nazwa:
        axs.set_title("osobno")
        naz = "osobno"
    elif "_r" in nazwa:
        axs.set_title("razem")
        naz = "razem"
    axs.axvline(np.mean(lista), label = "średnia histogramu", color = "k", linestyle="--")
    axs.hist(lista, color = 'cornflowerblue', label = 'losowe permuacje', alpha=0.7)
    #print(len(lista_imion1))
    #print(len(lista))
    for warunki, strs, color in zip(lista_org, ["nieznajomi", "znajomi", "pary"], ['g', 'c', 'k']):
        #plt.hist(lista, density=True)
        axs.axvline(np.mean(warunki), label = strs, color = color)
        with open(nazwa+czas+".txt", "a") as file1:
            # Writing data to a file
            file1.write("\n" + "\n" + strs + "\n")
            file1.writelines(str(stats.ttest_ind(a=lista, b=warunki)))

    axs.axvline(np.mean(lista_org), label = "cała grupa " + naz, color = 'r')
    with open(nazwa+czas+".txt", "a") as file1:
            # Writing data to a file
        file1.write("\n" + naz + "\n")
        file1.writelines(str(stats.ttest_ind(a=lista, b= np.concatenate(lista_org)))) 
    axs.legend()
for ax in axs2.flat:
    ax.label_outer()
plt.savefig("histogramy_per_sdnn")
plt.show()
fig, axs = plt.subplots(2, sharex=True, sharey=True)
axs[0].set_title("osobno")
axs[0].hist(lista_czas_nnstd_o)
axs[0].set_xlabel("przesunięcie czasowe dla max korelacji [s]")
axs[0].legend()
axs[1].set_title("razem")
axs[1].hist(lista_czas_nnstd_r)
axs[1].set_xlabel("przesunięcie czasowe dla max korelacji [s]")
axs[1].legend()
for ax in axs.flat:
    ax.label_outer()
plt.savefig("histogramy_czas_sdnn")
plt.show()"""
"""
show = False
for lista_o, lista_r, tytul, czas_max_corr_hrv_o, czas_max_corr_hrv_r in zip([lista_o_rmssd, lista_o_nnstd],[lista_r_rmssd, lista_r_nnstd],['rmssd.txt', "stdnn.txt"], [czas_max_corr_hrv_o, czas_nn_o], [czas_max_corr_hrv_r, czas_nn_r]):
    all_osobno = np.concatenate(lista_o)
    all_razem = np.concatenate(lista_r)
    #analysis.wykreslanka(all_osobno, all_razem, tytul)
    all_ = np.concatenate((all_osobno, all_razem))
    #rysowanie.minmax_plot(df_osobno, df_razem, all_razem, all_osobno, sek, tytul)
    rysowanie.histogram_ro(np.concatenate(czas_max_corr_hrv_o), np.concatenate(czas_max_corr_hrv_r))
    if show == True:
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
        analysis.normality_check(np.concatenate((lista_r[check],lista_o[check])))
    print("normalnosc wszysktich danych osobno")
    all_osobno1 = [i for i in all_osobno if i < 3]
    analysis.normality_check(all_osobno)
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
    print(aov)
    for oso, raz in zip(lista_o, lista_r):
        s, p = stats.ttest_ind(a=oso, b= raz)
        print(p)
    if show == True:
        pg.plot_paired(data=df, dv='korelacje', within='or', subject='subid')
        plt.tight_layout()
        plt.show()
    with open(tytul, "a") as file1:
        # Writing data to a file
        file1.writelines("\n" + "dla "+str(sek)+" sekundowego okienka" +"\n" )

        file1.writelines("r pearsona dla danych osobno " + str(stats.pearsonr(sur_ans_o, all_osobno, alternative='two-sided'))+"\n")
        file1.writelines("r pearsona dla danych razem "+ str(stats.pearsonr(sur_ans_r, all_razem, alternative='two-sided'))+"\n")
        file1.writelines("r pearsona dla danych razem-osobno "+ str(stats.pearsonr(sur_ans_r, all_razem-all_osobno, alternative='two-sided'))+"\n")
        file1.writelines("mixed_anova " + str(aov)+"\n") 
"""
"""fig, axs = plt.subplots(2, sharex=True, sharey=True)
for lista_o, lista_r, tytul, ax in zip([lista_o_rmssd,lista_o_nnstd], [lista_r_rmssd,lista_r_nnstd], ['RMSSD', "SDNN"], [axs[0],axs[1]]):
    rysowanie.znajomosc_scatter_ax(lista_o, lista_r, tytul, ax)

# Hide x labels and tick labels for top plots and y ticks for right plots
for ax in axs.flat:
    ax.label_outer()
plt.show()

fig, axs = plt.subplots(2, sharex=True, sharey=True)
for lista_o, lista_r, tytul, ax1 in zip([lista_o_rmssd ,lista_o_nnstd], [lista_r_rmssd,lista_r_nnstd], ['RMSSD', "SDNN"], [axs[0], axs[1]]):
    rysowanie.histogram_ro_ax(np.concatenate(lista_o), np.concatenate(lista_r),tytul, ax1)
    
for ax in axs.flat:
    ax.label_outer()
plt.show()


fig1, axs1 = plt.subplots(2, sharex=True, sharey=True)
for lista_r, tytul, ax1 in zip([lista_r_rmssd, lista_r_nnstd], ['RMSSD', "SDNN"], [axs1[0], axs1[1]]):
    rysowanie.scatter_urcs_znaj_ax(sur_ans_r, np.concatenate(lista_r),tytul, ax1)
    
for ax in axs1.flat:
    ax.label_outer()
plt.show()"""
lista_imion1 = np.concatenate(li)
lista = analysis.randomHR(df_osobno[2].iloc[3])
print(lista.shape)
print(np.concatenate(df_razem).shape)
lista_nnstd_r, lista_czas_nnstd_r = analysis.compars_f(lista_imion1, lista, sek, "nnstd_por")
plt.hist(lista_czas_nnstd_r)
plt.title("randowmowe dane")
plt.xlabel("przesunięcie czasowe dla max korelacji [s]")
plt.legend()
plt.savefig("histogramy_czasrandom_sdnn")
plt.show()
