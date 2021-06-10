globals().clear()

import pandas as pd
import matplotlib.pyplot as plt

# import numpy as np

# RELAZIONE TRA TENSIONE BATTERIA E VALORE LETTO DALL'ADC:
# Vbatt = 1881/69280 * ADC
param = 1881 / 69280
window_size = 50

data = pd.read_csv('durata_parte1.txt', sep=' ', header=None, engine='python')
data = pd.DataFrame(data)
data.columns = ['nans', 'seconds', 'ADC']
data = data.drop(['nans'], axis=1)

data_tomerge = pd.read_csv('BATT2.txt', sep=' ', header=None, engine='python')
data_tomerge = pd.DataFrame(data_tomerge)
data_tomerge.columns = ['nans', 'seconds', 'ADC']
data_tomerge= data_tomerge.drop(['nans'], axis=1)
data_tomerge['seconds'] = data_tomerge['seconds'].apply(lambda x: x + 280035)
print(data_tomerge)
data = data.append(data_tomerge)
print(data)

data['ADC'] = data['ADC'].astype(str)
data['ADC'] = data['ADC'].apply(int, base=16)

# for i in range(len(data)):
#    if data.iloc[i, 1] == 255:
#        data.iloc[i, 1] = np.nan
# data['ADC'].fillna(method='bfill', inplace=True)  # interpolate missing values
# data = data.reset_index(drop=True)

data['ADC'] = data['ADC'].apply(lambda x: x * param)

data['seconds'] = data['seconds'].apply(lambda x: x / 3600)  # converte in ore
print(data)
#  facciamo un po' di filtraggio perchè se no viene troppo ballerino
data['ADC'] = data['ADC'].rolling(window=window_size).sum()
data['ADC'] = data['ADC'] / window_size

plt.plot(data['seconds'], data['ADC'])
plt.ylim(top=2.65, bottom=1.5)
plt.title('Battery Voltage')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.show()
