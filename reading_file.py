import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pn


def wczytaj(path):
    try:
        # Read the CSV file with tab as the separator
        df = pd.read_csv(path, sep=",")
        df = df.T
        
        # Display the first few rows of the data
        data_razem_znajomi = df[(df[1]=='razem') & (df[2]=='znajomi')]
        data_razem_nieznajomi = df[(df[1]=='razem') & (df[2]=='nieznajomi')]
        data_razem_pary = df[(df[1]=='razem') & (df[2]=='para')]
        df_r = [data_razem_nieznajomi[data_razem_nieznajomi.columns[5:]].astype(float), data_razem_znajomi[data_razem_znajomi.columns[5:]].astype(float), data_razem_pary[data_razem_pary.columns[5:]].astype(float)]
        df_r_sur = [data_razem_nieznajomi[data_razem_nieznajomi.columns[3]].astype(float), data_razem_znajomi[data_razem_znajomi.columns[3]].astype(float), data_razem_pary[data_razem_pary.columns[3]].astype(float)]
        
        data_osobno_znajomi = df[(df[1]=='osobno') & (df[2]=='znajomi')]
        data_osobno_nieznajomi = df[(df[1]=='osobno') & (df[2]=='nieznajomi')]
        data_osobno_pary = df[(df[1]=='osobno') & (df[2]=='para')]
        df_o = [data_osobno_nieznajomi[data_osobno_nieznajomi.columns[5:]].astype(float), data_osobno_znajomi[data_osobno_znajomi.columns[5:]].astype(float), data_osobno_pary[data_osobno_pary.columns[5:]].astype(float)]
        df_o_sur = [data_razem_nieznajomi[data_razem_nieznajomi.columns[3]].astype(float), data_razem_znajomi[data_razem_znajomi.columns[3]].astype(float), data_razem_pary[data_razem_pary.columns[3]].astype(float)]
        return df_o, df_r, df_r_sur, df_o_sur
    except FileNotFoundError:
        print(f"File '{path}' not found. Please check the file path.")
        