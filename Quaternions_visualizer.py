globals().clear()
# PARAMETERS SELECTION
filename = 'paola14.txt'
win = 1200  #finestra di filtraggio
import pandas as pd
import matplotlib.pyplot as plt
import struct

def plotupdate():
    # CREAZIONE FINESTRA 1: QUATERNIONI E SEGNALE FILTRATO +PCA
    plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.title('Quaternions of device 1')
    plt.plot(data_1['1'], color='red')
    plt.plot(data_1['2'], color='green')
    plt.plot(data_1['3'], color='skyblue')
    plt.plot(data_1['4'], color='orange')
    plt.subplot(3, 1, 2)
    plt.title('Quaternions of device 2')
    plt.plot(data_2['1'], color='red')
    plt.plot(data_2['2'], color='green')
    plt.plot(data_2['3'], color='skyblue')
    plt.plot(data_2['4'], color='orange')
    plt.subplot(3, 1, 3)
    plt.title('Quaternions of device 3')
    plt.plot(data_3['1'], color='red')
    plt.plot(data_3['2'], color='green')
    plt.plot(data_3['3'], color='skyblue')
    plt.plot(data_3['4'], color='orange')
    return

data = pd.read_csv(filename, sep=",|:", header=None, engine='python')
data.columns = ['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'day', 'month', 'hour', 'min', 'sec', 'millisec']
data = data.reset_index(drop=True)  # reset the indexes order
print("Analizzo i dati a partire dal campione acquisito il giorno")
print(data.iloc[0, -6], "\\", data.iloc[0, -5], "alle", data.iloc[0, -4], ":",
      data.iloc[0, -3], ":", data.iloc[0, -2], ":", data.iloc[0, -1])
print("fino al campione acquisito il giorno")
print(data.iloc[-1, -6], "\\", data.iloc[-1, -5], "alle", data.iloc[-1, -4], ":",
          data.iloc[-1, -3], ":", data.iloc[-1, -2], ":", data.iloc[-1, -1])

fdev = 10
print("fdev:", round(fdev, 2), "Hz")
length = len(data)
print("Il dataset ha", length, "campioni")

# traforming into string in order to remove [ and ] from the file\
data['DevID'] = data['DevID'].astype(str)
data['DevID'] = data['DevID'].str.replace('[', '')
data['DevID'] = data['DevID'].str.replace(']', '')

data['1'] = data['1'].astype(str)
data['1'] = data['1'].str.replace('[', '')
data['1'] = data['1'].str.replace(']', '')

data['2'] = data['2'].astype(str)
data['2'] = data['2'].str.replace('[', '')
data['2'] = data['2'].str.replace(']', '')

data['3'] = data['3'].astype(str)
data['3'] = data['3'].str.replace('[', '')
data['3'] = data['3'].str.replace(']', '')

data['4'] = data['4'].astype(str)
data['4'] = data['4'].str.replace('[', '')
data['4'] = data['4'].str.replace(']', '')

data_1 = data[data['DevID'].str.contains('01')]  # thoracic data
data_1 = data_1.reset_index(drop=True)

data_2 = data[data['DevID'].str.contains('02')]  # abdominal data
data_2 = data_2.reset_index(drop=True)

data_3 = data[data['DevID'].str.contains('03')]  # reference data
data_3 = data_3.reset_index(drop=True)
# nth reception and information transmitted
data_1 = data_1[data_1.columns[3:8]]
data_2 = data_2[data_2.columns[3:8]]
data_3 = data_3[data_3.columns[3:8]]

data_1['1'] = data_1['1'].apply(int, base=16).rolling(window=win).sum()/win
data_2['1'] = data_2['1'].apply(int, base=16).rolling(window=win).sum()/win
data_3['1'] = data_3['1'].apply(int, base=16).rolling(window=win).sum()/win

data_1['2'] = data_1['2'].apply(int, base=16).rolling(window=win).sum()/win
data_2['2'] = data_2['2'].apply(int, base=16).rolling(window=win).sum()/win
data_3['2'] = data_3['2'].apply(int, base=16).rolling(window=win).sum()/win

data_1['3'] = data_1['3'].apply(int, base=16).rolling(window=win).sum()/win
data_2['3'] = data_2['3'].apply(int, base=16).rolling(window=win).sum()/win
data_3['3'] = data_3['3'].apply(int, base=16).rolling(window=win).sum()/win

data_1['4'] = data_1['4'].apply(int, base=16).rolling(window=win).sum()/win
data_2['4'] = data_2['4'].apply(int, base=16).rolling(window=win).sum()/win
data_3['4'] = data_3['4'].apply(int, base=16).rolling(window=win).sum()/win

#signed_value = ref.iloc[i, 1] / 127 if ref.iloc[i, 1] < 127 else (ref.iloc[i, 1] - 256) / 127

plotupdate()
plt.show()
