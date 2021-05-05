globals().clear()

digit2voltage = 9 / 640  # value used to convert sample value to voltage
window_size = 10  # number of samples taken for computing a chunk of data (600 = 1 minute of acquisition)
SgolayWindowPCA = 31
skip_chunks_start = 0  # number of intial chunks to skip
skip_chunks_end = 0  # number of final chunks to skip

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics
from pyquaternion import Quaternion
import numpy as np
import scipy.signal
from sklearn.decomposition import PCA
import scipy.stats as stats
import matplotlib.animation as animation

plt.rcParams.update({'figure.max_open_warning': 0})

data = pd.read_csv('test1.txt', sep=",|:", header=None, engine='python')
data.columns = ['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4']
data = data.reset_index(drop=True)  # reset the indexes order


# data.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data.csv', index = False, header=True)

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


# data of devices 1,2,3
tor = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
abd = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
ref = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])

pca = PCA(n_components=1)
t1 = pd.DataFrame(columns=['1', '2', '3', '4'])
tor_quat = Quaternion()
Tor_pose_quat = Quaternion()

ref_quat = Quaternion()
Ref_pose_quat = Quaternion()
FuseT_1 = []

index_data = 0  # global index for total data
count = 0
index_1, index_2, index_3 = 0, 0, 0  # indexes for devices
index_window = 0 # index for computing things in window
length = len(data)
print("Il dataset ha", length, "campioni")

#  PARTE ITERATIVA DEL CODICE

while index_data < length:
    print("INDEX:", index_data)
    # transforming into string in order to remove [ and ] from the file\
    data.iloc[index_data] = data.iloc[index_data].astype(str)
    data.iloc[index_data] = data.iloc[index_data].str.replace('[', '')
    data.iloc[index_data] = data.iloc[index_data].str.replace(']', '')
    data.iloc[index_data, 1:8] = data.iloc[index_data, 1:8].apply(int, base=16)  # convert to base 10 everything but DevID

    # Mette NAN ai quaternioni se il pacchetto è invalido
    if data.iloc[index_data, 2] == 255:  # 2 è la colonna C
        data.iloc[index_data, 4:8] = np.nan
        data.iloc[index_data, 1] = np.nan #mette nan anche al valore della batteria
        # print("Il nan è a", index)

    # Creazione dataframe del Reference (3)
    check = data.iloc[index_data].str.contains('03')
    if check['DevID'] == True:  # se device id è 3
        # mette il dato nel dataframe del terzo device
        ref = ref.append(data.iloc[index_data])
        ref = ref.reset_index(drop=True)
        ref = ref.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        ref = ref.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(3, index_3)  # device 3 conversion
        index_3 += 1

    # Creazione dataframe dell'addome (2)
    check = data.iloc[index_data].str.contains('2')
    if check['DevID'] == True:  # se device id è 2
        # mette il dato nel dataframe del terzo device
        abd = abd.append(data.iloc[index_data])
        abd = abd.reset_index(drop=True)
        abd = abd.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        abd = abd.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(2, index_2)  # device 1 conversion
        index_2 += 1

    # Creazione dataframe del torace (1)
    check = data.iloc[index_data].str.contains('01')
    if check['DevID'] == True:  # se device id è 1
        # mette il dato nel dataframe del terzo device
        tor = tor.append(data.iloc[index_data])
        tor = tor.reset_index(drop=True)
        tor = tor.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        tor = tor.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(1, index_1)  # device 1 conversion
        index_1 += 1

    if index_1 > window_size and index_2 > window_size and index_3 > window_size:

        # inizia a lavorare sui dati quando la prima finestra è piena
        #INTERPOLATE NON VA...fa divergere tutto
        #tor.interpolate(method='pchip', inplace=True)
        #abd.interpolate(method='pchip', inplace=True)
        #ref.interpolate(method='pchip', inplace=True)
        #tor = tor.loc[1:]
        tor.fillna(method='bfill', inplace=True)
        #tor = tor.reset_index(drop=True)
        #abd = abd.loc[1:]
        abd.fillna(method='bfill', inplace=True)
        #abd = abd.reset_index(drop=True)
        #ref = ref.loc[1:]
        ref.fillna(method='bfill', inplace=True)
        #ref = ref.reset_index(drop=True)

        index_window += 1

    index_data += 1  # global
    count += 1
    if count > 100:
        count = 0
        plt.subplot(3, 1, 1)
        plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
        plt.plot(tor[['1', '2', '3', '4']])
        plt.subplot(3, 1, 2)
        plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
        plt.plot(abd[['1', '2', '3', '4']])
        plt.subplot(3, 1, 3)
        plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
        plt.plot(ref[['1', '2', '3', '4']])
        plt.pause(0.01)


#plot eventually remaining data
plt.subplot(3, 1, 1)
plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
plt.plot(tor[['1', '2', '3', '4']])
plt.subplot(3, 1, 2)
plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
plt.plot(abd[['1', '2', '3', '4']])
plt.subplot(3, 1, 3)
plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
plt.plot(ref[['1', '2', '3', '4']])

plt.show()

#data1.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_1after.csv', index=False, header=True)
#data2.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_2after.csv', index=False, header=True)
#data3.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_3after.csv', index=False, header=True)
