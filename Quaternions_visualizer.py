globals().clear()
# PARAMETERS SELECTION
filename = 'ste13.txt'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plotupdate():
    # CREAZIONE FINESTRA 1: QUATERNIONI E SEGNALE FILTRATO +PCA
    plt.figure(1)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.title('Quaternions of device 1')
    plt.plot(tor['1'], color='red')
    plt.plot(tor['2'], color='green')
    plt.plot(tor['3'], color='skyblue')
    plt.plot(tor['4'], color='orange')
    plt.subplot(3, 1, 2)
    plt.title('Quaternions of device 2')
    plt.plot(abd['1'], color='red')
    plt.plot(abd['2'], color='green')
    plt.plot(abd['3'], color='skyblue')
    plt.plot(abd['4'], color='orange')
    plt.subplot(3, 1, 3)
    plt.title('Quaternions of device 3')
    plt.plot(ref['1'], color='red')
    plt.plot(ref['2'], color='green')
    plt.plot(ref['3'], color='skyblue')
    plt.plot(ref['4'], color='orange')
    plt.figure(2)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.title('Battery voltage of device 1')
    plt.plot(tor['B'].rolling(window=5).sum() / 5 * 1881 / 69280, color='red')
    plt.ylim(top=2.65, bottom=1.5)
    plt.subplot(3, 1, 2)
    plt.title('Battery voltage of device 2')
    plt.plot(abd['B'].rolling(window=5).sum() / 5 * 1881 / 69280, color='red')
    plt.ylim(top=2.65, bottom=1.5)
    plt.subplot(3, 1, 3)
    plt.title('Battery voltage of device 3')
    plt.plot(ref['B'].rolling(window=5).sum() / 5 * 1881 / 69280, color='red')
    plt.ylim(top=2.65, bottom=1.5)
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

length = len(data)
print("Il dataset ha", length, "campioni")
data = data.drop(['nthvalue'], axis=1)
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

data['B'] = data['B'].astype(str)
data['B'] = data['B'].str.replace('[', '')
data['B'] = data['B'].str.replace(']', '')

data['C'] = data['C'].astype(str)
data['C'] = data['C'].str.replace('[', '')
data['C'] = data['C'].str.replace(']', '')

tor = data[data['DevID'].str.contains('01')]  # thoracic data
tor = tor.reset_index(drop=True)

abd = data[data['DevID'].str.contains('02')]  # abdominal data
abd = abd.reset_index(drop=True)

ref = data[data['DevID'].str.contains('03')]  # reference data
ref = ref.reset_index(drop=True)
# nth reception and information transmitted
tor = tor[tor.columns[1:7]]
abd = abd[abd.columns[1:7]]
ref = ref[ref.columns[1:7]]

def quatconv(x):
    if x > 127 and x != np.nan:
        x -= 256
    x /= 127
    return x


tor['B'] = tor['B'].apply(int, base=16)
tor.loc[tor['C'] == 'FF', ['B']] = np.nan
tor['1'] = tor['1'].apply(int, base=16)
tor.loc[tor['C'] == 'FF', ['1']] = np.nan
tor['1'] = tor['1'].apply(lambda x: quatconv(x))
tor['2'] = tor['2'].apply(int, base=16)
tor.loc[tor['C'] == 'FF', ['2']] = np.nan
tor['2'] = tor['2'].apply(lambda x: quatconv(x))
tor['3'] = tor['3'].apply(int, base=16)
tor.loc[tor['C'] == 'FF', ['3']] = np.nan
tor['3'] = tor['3'].apply(lambda x: quatconv(x))
tor['4'] = tor['4'].apply(int, base=16)
tor.loc[tor['C'] == 'FF', ['4']] = np.nan
tor['4'] = tor['4'].apply(lambda x: quatconv(x))

abd['B'] = abd['B'].apply(int, base=16)
abd.loc[abd['C'] == 'FF', ['B']] = np.nan
abd['1'] = abd['1'].apply(int, base=16)
abd.loc[abd['C'] == 'FF', ['1']] = np.nan
abd['1'] = abd['1'].apply(lambda x: quatconv(x))
abd['2'] = abd['2'].apply(int, base=16)
abd.loc[abd['C'] == 'FF', ['2']] = np.nan
abd['2'] = abd['2'].apply(lambda x: quatconv(x))
abd['3'] = abd['3'].apply(int, base=16)
abd.loc[abd['C'] == 'FF', ['3']] = np.nan
abd['3'] = abd['3'].apply(lambda x: quatconv(x))
abd['4'] = abd['4'].apply(int, base=16)
abd.loc[abd['C'] == 'FF', ['4']] = np.nan
abd['4'] = abd['4'].apply(lambda x: quatconv(x))

ref['B'] = ref['B'].apply(int, base=16)
ref.loc[ref['C'] == 'FF', ['B']] = np.nan
ref['1'] = ref['1'].apply(int, base=16)
ref.loc[ref['C'] == 'FF', ['1']] = np.nan
ref['1'] = ref['1'].apply(lambda x: quatconv(x))
ref['2'] = ref['2'].apply(int, base=16)
ref.loc[ref['C'] == 'FF', ['2']] = np.nan
ref['2'] = ref['2'].apply(lambda x: quatconv(x))
ref['3'] = ref['3'].apply(int, base=16)
ref.loc[ref['C'] == 'FF', ['3']] = np.nan
ref['3'] = ref['3'].apply(lambda x: quatconv(x))
ref['4'] = ref['4'].apply(int, base=16)
ref.loc[ref['C'] == 'FF', ['4']] = np.nan
ref['4'] = ref['4'].apply(lambda x: quatconv(x))

#quaternioni belli belli, rimane solo da interpolarli
tor = tor.drop(['C'], axis=1)
abd = abd.drop(['C'], axis=1)
ref = ref.drop(['C'], axis=1)

tor.fillna(method='bfill', inplace=True)
tor = tor.reset_index(drop=True)
abd.fillna(method='bfill', inplace=True)
abd = abd.reset_index(drop=True)
ref.fillna(method='bfill', inplace=True)
ref = ref.reset_index(drop=True)

plotupdate()
plt.show()
