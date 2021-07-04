globals().clear()
# PARAMETERS SELECTION
filename = '3luglio.txt'
#A:sit.wo.su, B:sit, C:supine, D:prone, E:lyingL, F:lyingR, G:standing, I:stairs, L:walkS, M:walkF, N:run, O:cyclette
window_size = 600  # samples inside the window (Must be >=SgolayWindowPCA). Original: 97
SgolayWindowPCA = 31  # original: 31.  MUST BE AN ODD NUMBER
start = 0  # number of initial samples to skip (samples PER device) e.g.: 200 will skip 600 samples in total
stop = 0  # number of sample at which program execution will stop, 0 will run the whole txt file
incr = 300  # Overlapping between a window and the following. 1=max overlap. MUST BE >= SgolayWindowPCA. The higher the faster

import pandas as pd
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import numpy as np
from sklearn.decomposition import PCA
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)



def quatsconv(device, i):
    if device == 3:
        if ref.iloc[i, 1] > 127 and ref.iloc[i, 1] != np.nan:  # quat1 dev 3
            ref.iloc[i, 1] -= 256
            ref.iloc[i, 1] /= 127
        else:
            ref.iloc[i, 1] /= 127
        if ref.iloc[i, 2] > 127 and ref.iloc[i, 2] != np.nan:  # quat2 dev 3
            ref.iloc[i, 2] -= 256
            ref.iloc[i, 2] /= 127
        else:
            ref.iloc[i, 2] /= 127
        if ref.iloc[i, 3] > 127 and ref.iloc[i, 3] != np.nan:  # quat3 dev 3
            ref.iloc[i, 3] -= 256
            ref.iloc[i, 3] /= 127
        else:
            ref.iloc[i, 3] /= 127
        if ref.iloc[i, 4] > 127 and ref.iloc[i, 4] != np.nan:  # quat4 dev 3
            ref.iloc[i, 4] -= 256
            ref.iloc[i, 4] /= 127
        else:
            ref.iloc[i, 4] /= 127
    if device == 2:
        if abd.iloc[i, 1] > 127 and abd.iloc[i, 1] != np.nan:  # quat1 dev 3
            abd.iloc[i, 1] -= 256
            abd.iloc[i, 1] /= 127
        else:
            abd.iloc[i, 1] /= 127
        if abd.iloc[i, 2] > 127 and abd.iloc[i, 2] != np.nan:  # quat2 dev 3
            abd.iloc[i, 2] -= 256
            abd.iloc[i, 2] /= 127
        else:
            abd.iloc[i, 2] /= 127
        if abd.iloc[i, 3] > 127 and abd.iloc[i, 3] != np.nan:  # quat3 dev 3
            abd.iloc[i, 3] -= 256
            abd.iloc[i, 3] /= 127
        else:
            abd.iloc[i, 3] /= 127
        if abd.iloc[i, 4] > 127 and abd.iloc[i, 4] != np.nan:  # quat4 dev 3
            abd.iloc[i, 4] -= 256
            abd.iloc[i, 4] /= 127
        else:
            abd.iloc[i, 4] /= 127
    if device == 1:
        if tor.iloc[i, 1] > 127 and tor.iloc[i, 1] != np.nan:  # quat1 dev 1
            tor.iloc[i, 1] -= 256
            tor.iloc[i, 1] /= 127
        else:
            tor.iloc[i, 1] /= 127
        if tor.iloc[i, 2] > 127 and tor.iloc[i, 2] != np.nan:  # quat2 dev 1
            tor.iloc[i, 2] -= 256
            tor.iloc[i, 2] /= 127
        else:
            tor.iloc[i, 2] /= 127
        if tor.iloc[i, 3] > 127 and tor.iloc[i, 3] != np.nan:  # quat3 dev 1
            tor.iloc[i, 3] -= 256
            tor.iloc[i, 3] /= 127
        else:
            tor.iloc[i, 3] /= 127
        if tor.iloc[i, 4] > 127 and tor.iloc[i, 4] != np.nan:  # quat4 dev 1
            tor.iloc[i, 4] -= 256
            tor.iloc[i, 4] /= 127
        else:
            tor.iloc[i, 4] /= 127
    return


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
    return


data = pd.read_csv(filename, sep=",|:", header=None, engine='python')
#print(data)
data.columns = ['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'day', 'month', 'hour', 'min', 'sec', 'millisec']
data = data.reset_index(drop=True)  # reset the indexes order
#print(data)
print("Analizzo i dati a partire dal campione acquisito il giorno")
print(data.iloc[3*start, -6], "\\", data.iloc[3*start, -5], "alle", data.iloc[3*start, -4], ":",
      data.iloc[3*start, -3], ":", data.iloc[3*start, -2], ":", data.iloc[3*start, -1])
print("fino al campione acquisito il giorno")
if stop != 0:
    print(data.iloc[3*stop, -6], "\\", data.iloc[3*stop, -5], "alle", data.iloc[3*stop, -4], ":",
          data.iloc[3*stop, -3], ":", data.iloc[3*stop, -2], ":", data.iloc[3*stop, -1])
else:
    print(data.iloc[-1, -6], "\\", data.iloc[-1, -5], "alle", data.iloc[-1, -4], ":",
          data.iloc[-1, -3], ":", data.iloc[-1, -2], ":", data.iloc[-1, -1])

fdev = 10
print("fdev:", round(fdev, 2), "Hz")
# GLOBAL VARIABLES INITIALIZATION
tor = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
abd = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
ref = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
index_tor, index_abd, index_ref = 0, 0, 0  # indici per dataframe dei device
count = 0
index_window = 0  # for computing things inside the window
flag = 0  # used for plotting after first window is available

index_data = 3 * start  # global index for total data
length = len(data)
ncycles = length if stop == 0 else 3*stop
print("Il dataset ha", length, "campioni")
print("Skipping ", start, "data points")

#  PARTE ITERATIVA DEL CODICE
while index_data < ncycles:
    #print("GLOBAL INDEX:", index_data)
    # transforming into string in order to remove [ and ] from the file\
    data.iloc[index_data] = data.iloc[index_data].astype(str)
    data.iloc[index_data] = data.iloc[index_data].str.replace('[', '')
    data.iloc[index_data] = data.iloc[index_data].str.replace(']', '')
    data.iloc[index_data, 1:8] = data.iloc[index_data, 1:8].apply(int,
                                                                  base=16)  # convert to base 10 everything but DevID and timestamp

    # Mette NAN ai quaternioni se il pacchetto è invalido
    if data.iloc[index_data, 2] == 255:  # 2 è la colonna C
        data.iloc[index_data, 4:8] = np.nan
        data.iloc[index_data, 1] = np.nan  # mette nan anche al valore della batteria (colonna B)
        # print("Il nan è a", index_data, "ed è il device", data.iloc[index_data, 0])

    # Reference (3) dataframe extension
    check = data.iloc[index_data].str.contains('03')
    if check['DevID']:  # se device id è 3
        # mette il dato nel dataframe del terzo device
        ref = ref.append(data.iloc[index_data])
        ref = ref.reset_index(drop=True)
        ref = ref.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery, timestamp and quaternions data
        ref = ref.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(3, index_ref)  # device 3 conversion
        index_ref += 1
    # Abdomen (2) dataframe extension
    check = data.iloc[index_data].str.contains('02')
    if check['DevID']:  # se device id è 2
        # mette il dato nel dataframe del terzo device
        abd = abd.append(data.iloc[index_data])
        abd = abd.reset_index(drop=True)
        abd = abd.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery, timestamp and quaternions data
        abd = abd.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(2, index_abd)  # device 1 conversion
        index_abd += 1
        # print(abd)
    # Thorax (1) dataframe extension
    check = data.iloc[index_data].str.contains('01')
    if check['DevID']:  # se device id è 1
        # mette il dato nel dataframe del terzo device
        tor = tor.append(data.iloc[index_data])
        tor = tor.reset_index(drop=True)
        tor = tor.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery, timestamp and quaternions data
        tor = tor.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(1, index_tor)  # device 1 conversion
        index_tor += 1

    index_data += 1  # global
    count += 1
    if count > 1000:
        count = 0
        plotupdate()
        plt.pause(0.01)
print("END")
plt.show()
