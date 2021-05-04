import pandas as pd

data = pd.read_csv('Stefano_L_A.txt', sep=",|:", header=None, engine='python')
data.columns = ['TxRx', 'DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'None']
# select only the Rx line
data = data.loc[data['TxRx'] == 'Rx']
data = data.drop(['TxRx', 'None'], axis=1)
data = data.reset_index(drop=True)  # reset the indexes order
data['DevID'] = data['DevID'].astype(str)
length = len(data)
for i in range(length):

    if data.iloc[i, 0] == ' [01]':  #se DevID è 1
        last1 = data.iloc[i, 3]
        if data.iloc[i, 3] != last1+1:
            empty_row = pd.DataFrame([], index=[])  # creating the empty row
            data_1 = pd.concat([data_1.loc[:j + i * 256 - 1], empty_row, data_1.loc[j + i * 256:]])
        print(last1)

#data.to_csv (r'C:\Users\Stefano\Desktop\Analisi del segnale\out.csv', index = False, header=True)
