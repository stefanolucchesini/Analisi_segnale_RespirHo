import pandas as pd
import numpy as np


data = pd.read_csv('Stefano_L_A.txt', sep=",|:", header=None, engine='python')
data.columns = ['TxRx', 'DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'None']
# select only the Rx line
data = data.loc[data['TxRx'] == 'Rx']
data = data.drop(['TxRx', 'None'], axis=1)
data['nthvalue'] = data['nthvalue'].astype(str)
data['nthvalue'] = data['nthvalue'].str.replace('[', '')
data['nthvalue'] = data['nthvalue'].str.replace(']', '')
data['nthvalue'] = data['nthvalue'].apply(int, base=16)
data['nthvalue'] = data['nthvalue'].astype(int)
data = data.reset_index(drop=True)  # reset the indexes order
data['DevID'] = data['DevID'].astype(str)
print(data)
last1 = 0
for i in range(len(data)):
    print("i:", i)
    if data.iloc[i, 0] == ' [01]':  #se DevID è 1
        if data.iloc[i, 3] > last1+1:  #se nthvalue è più grande del precedente di più di una unità
            for u in range(0, data.iloc[i, 3] - last1 - 1):
                print("u:", u)
                if last1+1+u > 255:
                    next = last1+1+u - 256
                else:
                    next = last1+1+u
                empty_row = pd.DataFrame({'DevID': " [01]", "B": "[00]", "C": "[FF]", "nthvalue": next, "1": "[00]", "2": "[00]", "3": "[00]", "4": "[00]"}, index=[i - (data.iloc[i, 3] - last1 - 1) + u])  # creating the empty row
                print("LET ME ADD AN EMPTY ROW:\n", empty_row)
                print("BEFORE ADDING\n", data.iloc[:i + u+ 2].tail(10))
                data = pd.concat([data.loc[:i - (data.iloc[i, 3] - last1 - 1) + u], empty_row, data.loc[i - (data.iloc[i, 3] - last1 - 1) + u:]]).reset_index(drop=True)
                print("AFTER ADDING\n", data.iloc[:i+u+5].tail(10))
            last1 = data.iloc[i, 3]
        else:
            last1 = data.iloc[i, 3]
data.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\out.csv', index = False, header=True)
