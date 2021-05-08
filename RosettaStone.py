import pandas as pd
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('Stefano_L_D.txt', sep=",|:", header=None, engine='python')
data.columns = ['TxRx', 'DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'None']
# select only the Rx line
data = data.loc[data['TxRx'] == 'Rx']
data = data.drop(['TxRx', 'None'], axis=1)
data = data.reset_index(drop=True)  # reset the indexes order

data['nthvalue'] = data['nthvalue'].astype(str)
data['nthvalue'] = data['nthvalue'].str.replace('[', '')
data['nthvalue'] = data['nthvalue'].str.replace(']', '')

data_1 = data[data['DevID'].str.contains('01')]  # thoracic data
data_1 = data_1.reset_index(drop=True)

data_2 = data[data['DevID'].str.contains('02')]  # abdominal data
data_2 = data_2.reset_index(drop=True)

data_3 = data[data['DevID'].str.contains('03')]  # reference data
data_3 = data_3.reset_index(drop=True)

data_1_len_original = len(data_1)
data_2_len_original = len(data_2)
data_3_len_original = len(data_3)

print("Numero di campioni acquisiti da unita' 1: ", data_1_len_original)
print("Numero di campioni acquisiti da unita' 2: ", data_2_len_original)
print("Numero di campioni acquisiti da unita' 3: ", data_3_len_original)

if len(data_3) > len(data_1) and len(data_3) > len(data_2):
    max_value = math.floor(len(data_3) / 256)
elif len(data_1) > len(data_2) and len(data_1) > len(data_3):
    max_value = math.floor(len(data_1) / 256)
else:
    max_value = math.floor((len(data_2)) / 256)

# conversion from hexadecimal to decimal
data_1['nthvalue'] = data_1['nthvalue'].apply(int, base=16)
data_2['nthvalue'] = data_2['nthvalue'].apply(int, base=16)
data_3['nthvalue'] = data_3['nthvalue'].apply(int, base=16)

max_value = max_value - 1

dev_1_losses, dev_2_losses, dev_3_losses = 0, 0, 0

# add an empty row when there is a "jump" in communication, when the nth value is not received
# data_1.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_1bef.csv', index=False, header=True)
print("\nAggiungo dati mancanti invalidati")

for i in range(max_value):
    for j in range(256):
        if data_1['nthvalue'][j + i * 256] != j:
            dev_1_losses += 1
            empty_row = pd.DataFrame(
                {'DevID': " [01]", "B": "[00]", "C": "[FF]", "nthvalue": j, "1": "[00]", "2": "[00]", "3": "[00]",
                 "4": "[00]"}, index=[j + i * 256])  # creating the empty row
            data_1 = pd.concat([data_1.loc[:j + i * 256 - 1], empty_row, data_1.loc[j + i * 256:]])
            # print(i)
            data_1 = data_1.reset_index(drop=True)
data_1 = data_1.iloc[:max_value * 256]

data_2.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_2bef.csv', index=False, header=True)
for i in range(max_value):
    for j in range(256):
        if data_2['nthvalue'][j + i * 256] != j:
            dev_2_losses += 1
            empty_row = pd.DataFrame(
                {'DevID': " [02]", "B": "[00]", "C": "[FF]", "nthvalue": j, "1": "[00]", "2": "[00]", "3": "[00]",
                 "4": "[00]"}, index=[j + i * 256])  # creating the empty data
            data_2 = pd.concat([data_2.loc[:j + i * 256 - 1], empty_row, data_2.loc[j + i * 256:]])
            data_2 = data_2.reset_index(drop=True)
data_2 = data_2.iloc[:max_value * 256]

for i in range(max_value):
    for j in range(256):
        if data_3['nthvalue'][j + i * 256] != j:
            dev_3_losses += 1
            empty_row = pd.DataFrame(
                {'DevID': " [03]", "B": "[00]", "C": "[FF]", "nthvalue": j, "1": "[00]", "2": "[00]", "3": "[00]",
                 "4": "[00]"}, index=[j + i * 256])  # creating the empty data
            data_3 = pd.concat([data_3.loc[:j + i * 256 - 1], empty_row, data_3.loc[j + i * 256:]])
            data_3 = data_3.reset_index(drop=True)
data_3 = data_3.iloc[:max_value * 256]

data_1['nthvalue'] = data_1['nthvalue'].apply(hex)
data_1['nthvalue'] = data_1['nthvalue'].str.replace('0x', '[')
data_1['nthvalue'] = data_1['nthvalue'] + ']'
data_2['nthvalue'] = data_2['nthvalue'].apply(hex)
data_2['nthvalue'] = data_2['nthvalue'].str.replace('0x', '[')
data_2['nthvalue'] = data_2['nthvalue'] + ']'
data_3['nthvalue'] = data_3['nthvalue'].apply(hex)
data_3['nthvalue'] = data_3['nthvalue'].str.replace('0x', '[')
data_3['nthvalue'] = data_3['nthvalue'] + ']'

data_1_len = len(data_1)
data_2_len = len(data_2)
data_3_len = len(data_3)

print("\nNumero di campioni finali da unita' 1:", data_1_len, "\nCampioni persi", dev_1_losses, "\nData loss:", math.ceil(dev_1_losses/data_1_len_original*100), "%\n")
print("Numero di campioni finali da unita' 2:", data_2_len, "\nCampioni persi", dev_2_losses, "\nData loss:", math.ceil(dev_2_losses/data_2_len_original*100), "%\n")
print("Numero di campioni finali da unita' 3:", data_3_len, "\nCampioni persi", dev_3_losses, "\nData loss:", math.ceil(dev_3_losses/data_3_len_original*100), "%\n")

# data_1.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_1after.csv', index=False, header=True)
data_2.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_2after.csv', index=False, header=True)
# data_3.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_3after.csv', index=False, header=True)
print("Il data loss tipico è intorno al 20%\nCreo dataframe nel nuovo formato...")
converted_df = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
if data_1_len == data_2_len and data_3_len == data_2_len:
    for i in range(data_1_len):
        converted_df = converted_df.append(data_1.iloc[i])
        converted_df = converted_df.append(data_2.iloc[i])
        converted_df = converted_df.append(data_3.iloc[i])
else:
    print("ERRORE! I 3 DATAFRAME NON HANNO LA SOLITA LUNGHEZZA")

print(converted_df)
converted_df = converted_df.reset_index(drop=True)  # reset the indexes order
converted_df.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\Stefano_L_D_new.txt', header=None, index=None,
                    sep=',')
print("FINE")
