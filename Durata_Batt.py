globals().clear()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#RELAZIONE TRA TENSIONE BATTERIA E VALORE LETTO DALL'ADC:
#Vbatt = 1881/69280 * ADC
param = 1881/69280

data = pd.read_csv('batt.txt', sep=' ', header=None, engine='python')
data = pd.DataFrame(data)
data.columns = ['nans', 'seconds', 'ADC']
data = data.drop(['nans'], axis=1)

data['ADC'] = data['ADC'].astype(str)
data['ADC'] = data['ADC'].apply(int, base=16)
for i in range(len(data)):
    if data.iloc[i, 1] == 255:
        data.iloc[i, 1] = np.nan

data['ADC'].fillna(method='bfill', inplace=True)  # interpolate missing values
data = data.reset_index(drop=True)
data['ADC'] = data['ADC'].apply(lambda x: x*param)

data['seconds'] = data['seconds'].apply(lambda x: x/60)  #converte in minuti
plt.plot(data['seconds'], data['ADC'], label='batt voltage')
plt.title('Battery voltage')
plt.xlabel('Time (minutes)')
plt.ylabel('Voltage (V)')
plt.show()

