import reading_file
import analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
import pingouin as pg
import rysowanie


df_osobno, df_razem, df_r_sur, df_o_sur = reading_file.wczytaj("pomiary_licencjat_ost.csv")
sur_ans_o = np.concatenate(df_o_sur)
sur_ans_o = [i for i in sur_ans_o[::2]]
sur_ans_r = np.concatenate(df_r_sur)
sur_ans_r = [i for i in sur_ans_r[::2]]

lista_o = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r = np.zeros((len(df_razem), df_razem[0].shape[0]//2))
lista_o_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))
lista_r_rmssd = np.zeros((len(df_osobno), df_osobno[0].shape[0]//2))

for i in range(len(df_osobno)):
    for k, i_o in enumerate(range(0, df_osobno[0].shape[0]-1, 2)):
        df_osobno0 = analysis.standarize_data(df_osobno[i].iloc[i_o])
        df_osobno1 = analysis.standarize_data(df_osobno[i].iloc[i_o+1])
        df_o_intpol0 = analysis.interpolate_data(df_osobno0) 
        df_o_intpol1 = analysis.interpolate_data(df_osobno1)
        lista_o[i, k] = analysis.rolling_corr(df_o_intpol0, df_o_intpol1)
        lista_o_rmssd[i,k] = analysis.rmssd_corr(df_o_intpol0, df_o_intpol1, 60)

        df_razem0 = analysis.standarize_data(df_razem[i].iloc[i_o])
        df_razem1 = analysis.standarize_data(df_razem[i].iloc[i_o+1])
        df_r_intpol0 = analysis.interpolate_data(df_razem0) 
        df_r_intpol1 = analysis.interpolate_data(df_razem1)
        lista_r[i, k] = analysis.rolling_corr(df_r_intpol0, df_r_intpol1)
        lista_r_rmssd[i,k] = analysis.rmssd_corr(df_r_intpol0, df_r_intpol1, 60)

for lista_o, lista_r, tytul in zip([lista_o, lista_o_rmssd],[lista_r, lista_r_rmssd],['HR','HRV-RMSSD']):
    all_osobno = np.concatenate(lista_o)
    all_razem = np.concatenate(lista_r)
    all_ = np.concatenate((all_osobno, all_razem))

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
    print(f_oneway(lista_o[0], lista_o[1], lista_o[2]))

    print(f_oneway(lista_r[0], lista_r[1], lista_r[2]))

    print(f_oneway(roznice[0], roznice[1], roznice[2]))

    print(stats.ttest_ind(a=all_osobno, b=all_razem, equal_var=True, alternative='less'))

    razem_relacja = np.concatenate((lista_r[1], lista_r[2]))
    print(stats.ttest_ind(a=razem_relacja, b=lista_r[0], equal_var=True, alternative='greater'))

    relacja_roznice = np.concatenate((roznice[1], roznice[2]))
    print(stats.ttest_ind(a=relacja_roznice, b=roznice[0], equal_var=True, alternative='greater'))


    print("sprawdzanie normalności danych")
    for check in range(0,3):
        print("razem" + str(check))
        analysis.normality_check(lista_r[check])
    for check in range(0,3):
        print("osobno" + str(check))
        analysis.normality_check(lista_o[check])


"""result = [0 for i in range(all_osobno.shape[0])] + [1 for i in range(all_razem.shape[0])]
znajomosci = [0 for i in range(lista_o[0].shape[0])] + [1 for i in range(lista_o[1].shape[0])]+ [2 for i in range(lista_o[2].shape[0])]+ [0 for i in range(lista_r[0].shape[0])]+ [1 for i in range(lista_r[1].shape[0])]+ [2 for i in range(lista_r[2].shape[0])]

subjects = list(range(1, len(all_)//2+1))+list(range(1, len(all_)//2+1))

df = pd.DataFrame({
    "subject_id": subjects,
  "korelacje": all_,
  "osobno_razem": result,
  "znajomosci": znajomosci
})

aov = pg.rm_anova(data=df, dv="korelacje", within=["osobno_razem", "znajomosci"], subject='subject_id', detailed=True)
print(aov)"""





    



