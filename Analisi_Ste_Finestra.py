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
        if data3.iloc[i, 1] > 127 and data3.iloc[i, 1] != np.nan:  # quat1 dev 3
            data3.iloc[i, 1] -= 256
            data3.iloc[i, 1] /= 127
        else:
            data3.iloc[i, 1] /= 127
        if data3.iloc[i, 2] > 127 and data3.iloc[i, 2] != np.nan:  # quat2 dev 3
            data3.iloc[i, 2] -= 256
            data3.iloc[i, 2] /= 127
        else:
            data3.iloc[i, 2] /= 127
        if data3.iloc[i, 3] > 127 and data3.iloc[i, 3] != np.nan:  # quat3 dev 3
            data3.iloc[i, 3] -= 256
            data3.iloc[i, 3] /= 127
        else:
            data3.iloc[i, 3] /= 127
        if data3.iloc[i, 4] > 127 and data3.iloc[i, 4] != np.nan:  # quat4 dev 3
            data3.iloc[i, 4] -= 256
            data3.iloc[i, 4] /= 127
        else:
            data3.iloc[i, 4] /= 127
    if device == 2:
        if data2.iloc[i, 1] > 127 and data2.iloc[i, 1] != np.nan:  # quat1 dev 3
            data2.iloc[i, 1] -= 256
            data2.iloc[i, 1] /= 127
        else:
            data2.iloc[i, 1] /= 127
        if data2.iloc[i, 2] > 127 and data2.iloc[i, 2] != np.nan:  # quat2 dev 3
            data2.iloc[i, 2] -= 256
            data2.iloc[i, 2] /= 127
        else:
            data2.iloc[i, 2] /= 127
        if data2.iloc[i, 3] > 127 and data2.iloc[i, 3] != np.nan:  # quat3 dev 3
            data2.iloc[i, 3] -= 256
            data2.iloc[i, 3] /= 127
        else:
            data2.iloc[i, 3] /= 127
        if data2.iloc[i, 4] > 127 and data2.iloc[i, 4] != np.nan:  # quat4 dev 3
            data2.iloc[i, 4] -= 256
            data2.iloc[i, 4] /= 127
        else:
            data2.iloc[i, 4] /= 127
    if device == 1:
        if data1.iloc[i, 1] > 127 and data1.iloc[i, 1] != np.nan:  # quat1 dev 1
            data1.iloc[i, 1] -= 256
            data1.iloc[i, 1] /= 127
        else:
            data1.iloc[i, 1] /= 127
        if data1.iloc[i, 2] > 127 and data1.iloc[i, 2] != np.nan:  # quat2 dev 1
            data1.iloc[i, 2] -= 256
            data1.iloc[i, 2] /= 127
        else:
            data1.iloc[i, 2] /= 127
        if data1.iloc[i, 3] > 127 and data1.iloc[i, 3] != np.nan:  # quat3 dev 1
            data1.iloc[i, 3] -= 256
            data1.iloc[i, 3] /= 127
        else:
            data1.iloc[i, 3] /= 127
        if data1.iloc[i, 4] > 127 and data1.iloc[i, 4] != np.nan:  # quat4 dev 1
            data1.iloc[i, 4] -= 256
            data1.iloc[i, 4] /= 127
        else:
            data1.iloc[i, 4] /= 127
    return


# data of devices 1,2,3
data1 = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
data2 = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])
data3 = pd.DataFrame(columns=['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4'])

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

    if data.iloc[index_data, 2] == 255:  # 2 è la colonna B
        data.iloc[index_data, 4:8] = np.nan
        # print("Il nan è a", index)

    # Creazione dataframe del Reference (3)
    check = data.iloc[index_data].str.contains('03')
    if check['DevID'] == True:  # se device id è 3
        # mette il dato nel dataframe del terzo device
        data3 = data3.append(data.iloc[index_data])
        data3 = data3.reset_index(drop=True)
        data3 = data3.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        # conversion of quaternions in range [-1:1]
        quatsconv(3, index_3)  # device 3 conversion
        index_3 += 1



    # Creazione dataframe dell'addome (2)
    check = data.iloc[index_data].str.contains('2')
    if check['DevID'] == True:  # se device id è 2
        # mette il dato nel dataframe del terzo device
        data2 = data2.append(data.iloc[index_data])
        data2 = data2.reset_index(drop=True)
        data2 = data2.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        # conversion of quaternions in range [-1:1]
        quatsconv(2, index_2)  # device 1 conversion
        index_2 += 1

    # Creazione dataframe del torace (1)
    check = data.iloc[index_data].str.contains('01')
    if check['DevID'] == True:  # se device id è 1
        # mette il dato nel dataframe del terzo device
        data1 = data1.append(data.iloc[index_data])
        data1 = data1.reset_index(drop=True)
        data1 = data1.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        # conversion of quaternions in range [-1:1]
        quatsconv(1, index_1)  # device 1 conversion
        index_1 += 1

    if index_1 > window_size and index_2 > window_size and index_3 > window_size:

        # inizia a lavorare sui dati quando la prima finestra è piena
        #REFERENCE
        #data3['1'] = data3['1'].interpolate(method='pchip')  rischio divergenza se mancano tanti dati!!
        data3['1'] = data3['1'].fillna(method='bfill')
        #data3['2'] = data3['2'].interpolate(method='pchip')
        data3['2'] = data3['2'].fillna(method='bfill')
        #data3['3'] = data3['3'].interpolate(method='pchip')
        data3['3'] = data3['3'].fillna(method='bfill')
        #data3['4'] = data3['4'].interpolate(method='pchip')
        data3['4'] = data3['4'].fillna(method='bfill')
        #ABDOMEN
        #data2['1'] = data2['1'].interpolate(method='pchip')
        data2['1'] = data2['1'].fillna(method='bfill')
        #data2['2'] = data2['2'].interpolate(method='pchip')
        data2['2'] = data2['2'].fillna(method='bfill')
        #data2['3'] = data2['3'].interpolate(method='pchip')
        data2['3'] = data2['3'].fillna(method='bfill')
        #data2['4'] = data2['4'].interpolate(method='pchip')
        data2['4'] = data2['4'].fillna(method='bfill')
        #THORAX
        #data1['1'] = data1['1'].interpolate(method='pchip')
        data1['1'] = data1['1'].fillna(method='bfill')
        #data1['2'] = data1['2'].interpolate(method='pchip')
        data1['2'] = data1['2'].fillna(method='bfill')
        #data1['3'] = data1['3'].interpolate(method='pchip')
        data1['3'] = data1['3'].fillna(method='bfill')
        #data1['4'] = data1['4'].interpolate(method='pchip')
        data1['4'] = data1['4'].fillna(method='bfill')

        index_window += 1



    index_data += 1  # global
    count += 1
    if count > 100:
        count = 0
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
        plt.ylim([-1, 1])
        plt.plot(data1['1'], color='red')
        plt.plot(data1['2'], color='green')
        plt.plot(data1['3'], color='skyblue')
        plt.plot(data1['4'], color='darkviolet')

        plt.subplot(3, 1, 2)
        plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
        plt.ylim([-1, 1])
        plt.plot(data2['1'], color='red')
        plt.plot(data2['2'], color='green')
        plt.plot(data2['3'], color='skyblue')
        plt.plot(data2['4'], color='darkviolet')
        plt.draw()

        plt.subplot(3, 1, 3)
        plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
        plt.ylim([-1, 1])
        plt.plot(data3['1'], color='red')
        plt.plot(data3['2'], color='green')
        plt.plot(data3['3'], color='skyblue')
        plt.plot(data3['4'], color='darkviolet')
        plt.pause(0.01)


#plot eventually remaining data
plt.clf()
plt.subplot(3, 1, 1)
plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
plt.ylim([-1, 1])
plt.plot(data1['1'], color='red')
plt.plot(data1['2'], color='green')
plt.plot(data1['3'], color='skyblue')
plt.plot(data1['4'], color='darkviolet')

plt.subplot(3, 1, 2)
plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
plt.ylim([-1, 1])
plt.plot(data2['1'], color='red')
plt.plot(data2['2'], color='green')
plt.plot(data2['3'], color='skyblue')
plt.plot(data2['4'], color='darkviolet')
plt.draw()

plt.subplot(3, 1, 3)
plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
plt.ylim([-1, 1])
plt.plot(data3['1'], color='red')
plt.plot(data3['2'], color='green')
plt.plot(data3['3'], color='skyblue')
plt.plot(data3['4'], color='darkviolet')
plt.show()

data1.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_1after.csv', index=False, header=True)
data2.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_2after.csv', index=False, header=True)
data3.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_3after.csv', index=False, header=True)
