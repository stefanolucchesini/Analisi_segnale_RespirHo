globals().clear()

digit2voltage = 9 / 640  # value used to convert sample value to voltage
chunksize = 128  # number of samples taken for computing a chunk of data

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

data = pd.read_csv('Stefano_L_C.txt', sep=",|:", header=None, engine='python')

data.columns = ['TxRx', 'DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'None']
# select only the Rx line
data = data.loc[data['TxRx'] == 'Rx']
data = data.drop(['TxRx', 'None'], axis=1)
data = data.reset_index(drop=True)  # reset the indexes order
# traforming into string in order to remove [ and ] from the file\
data['DevID'] = data['DevID'].astype(str)
data['DevID'] = data['DevID'].str.replace('[', '')
data['DevID'] = data['DevID'].str.replace(']', '')

data['B'] = data['B'].astype(str)
data['B'] = data['B'].str.replace('[', '')
data['B'] = data['B'].str.replace(']', '')

data['C'] = data['C'].astype(str)
data['C'] = data['C'].str.replace('[', '')
data['C'] = data['C'].str.replace(']', '')

data['nthvalue'] = data['nthvalue'].astype(str)
data['nthvalue'] = data['nthvalue'].str.replace('[', '')
data['nthvalue'] = data['nthvalue'].str.replace(']', '')

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

data_1 = data_1[data_1.columns[3:8]].copy()
data_2 = data_2[data_2.columns[3:8]].copy()
data_3 = data_3[data_3.columns[3:8]].copy()

print("Numero di campioni acquisiti da unita' 1: ", len(data_1))
print("Numero di campioni acquisiti da unita' 2: ", len(data_2))
print("Numero di campioni acquisiti da unita' 3: ", len(data_3))

if len(data_3) > len(data_1) and len(data_3) > len(data_2):
    max_value = math.floor(len(data_3) / 256)
    # max_value = math.ceil(len(data_3) / 256)
elif len(data_1) > len(data_2) and len(data_1) > len(data_3):
    max_value = math.floor(len(data_1) / 256)
    # max_value = math.ceil(len(data_1) / 256)
else:
    max_value = math.floor((len(data_2)) / 256)
# conversion from hexadecimal to decimal
data_1['nthvalue'] = data_1['nthvalue'].apply(int, base=16)
data_2['nthvalue'] = data_2['nthvalue'].apply(int, base=16)
data_3['nthvalue'] = data_3['nthvalue'].apply(int, base=16)

data_1['1'] = data_1['1'].apply(int, base=16)
data_2['1'] = data_2['1'].apply(int, base=16)
data_3['1'] = data_3['1'].apply(int, base=16)

data_1['2'] = data_1['2'].apply(int, base=16)
data_2['2'] = data_2['2'].apply(int, base=16)
data_3['2'] = data_3['2'].apply(int, base=16)

data_1['3'] = data_1['3'].apply(int, base=16)
data_2['3'] = data_2['3'].apply(int, base=16)
data_3['3'] = data_3['3'].apply(int, base=16)

data_1['4'] = data_1['4'].apply(int, base=16)
data_2['4'] = data_2['4'].apply(int, base=16)
data_3['4'] = data_3['4'].apply(int, base=16)

max_value = max_value - 1

for i in range(max_value):
    for j in range(256):
        if data_1['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty row
            data_1 = pd.concat([data_1.loc[:j + i * 256 - 1], empty_row, data_1.loc[j + i * 256:]])
            # print(i)
            data_1 = data_1.reset_index(drop=True)

data_1 = data_1.iloc[:max_value * 256]

for i in range(max_value):
    for j in range(256):
        if data_2['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty data
            data_2 = pd.concat([data_2.loc[:j + i * 256 - 1], empty_row, data_2.loc[j + i * 256:]])
            data_2 = data_2.reset_index(drop=True)

data_2 = data_2.iloc[:max_value * 256]

for i in range(max_value):
    for j in range(256):
        if data_3['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty data
            data_3 = pd.concat([data_3.loc[:j + i * 256 - 1], empty_row, data_3.loc[j + i * 256:]])
            data_3 = data_3.reset_index(drop=True)

data_3 = data_3.iloc[:max_value * 256]

list1 = [len(data_1), len(data_2), len(data_3)]
min_samples = min(list1)  # shortest number of samples
n_chunks = math.floor(min_samples / chunksize)
print("Divido il dataset in", n_chunks, "parti da", chunksize, "campioni l'una")

# cut every dataframe to the size
data_1 = data_1[:n_chunks * chunksize]
data_2 = data_2[:n_chunks * chunksize]
data_3 = data_3[:n_chunks * chunksize]

tor_quat = Quaternion()
Tor_pose_quat = Quaternion()
tor_pose = pd.DataFrame(columns=['1', '2', '3', '4'])

for c in range(n_chunks):
    pezzo11 = data_1.loc[c * chunksize:(c + 1) * chunksize - 1, '1']
    pezzo11.interpolate(method='pchip', inplace=True)
    pezzo11.fillna(method='bfill', inplace=True)

    pezzo12 = data_1.loc[c * chunksize:(c + 1) * chunksize - 1, '2']
    pezzo12.interpolate(method='pchip', inplace=True)
    pezzo12.fillna(method='bfill', inplace=True)

    pezzo13 = data_1.loc[c * chunksize:(c + 1) * chunksize - 1, '3']
    pezzo13.interpolate(method='pchip', inplace=True)
    pezzo13.fillna(method='bfill', inplace=True)

    pezzo14 = data_1.loc[c * chunksize:(c + 1) * chunksize - 1, '4']
    pezzo14.interpolate(method='pchip', inplace=True)
    pezzo14.fillna(method='bfill', inplace=True)

    mean11 = 0
    mean12 = 0
    mean13 = 0
    mean14 = 0

    for i in range(chunksize):
        if pezzo11[i + c * chunksize] > 127:
            pezzo11[i + c * chunksize] -= 256
            pezzo11[i + c * chunksize] /= 127
        else:
            pezzo11[i + c * chunksize] /= 127
        mean11 += pezzo11[i + c * chunksize]
        if pezzo12[i + c * chunksize] > 127:
            pezzo12[i + c * chunksize] -= 256
            pezzo12[i + c * chunksize] /= 127
        else:
            pezzo12[i + c * chunksize] /= 127
        mean12 += pezzo12[i + c * chunksize]
        if pezzo13[i + c * chunksize] > 127:
            pezzo13[i + c * chunksize] -= 256
            pezzo13[i + c * chunksize] /= 127
        else:
            pezzo13[i + c * chunksize] /= 127
        mean13 += pezzo13[i + c * chunksize]
        if pezzo14[i + c * chunksize] > 127:
            pezzo14[i + c * chunksize] -= 256
            pezzo14[i + c * chunksize] /= 127
        else:
            pezzo14[i + c * chunksize] /= 127
        mean14 += pezzo14[i + c * chunksize]
        #qua dentro calcola il quaternione
        tor_quat = Quaternion(pezzo11[i + c * chunksize], pezzo12[i + c * chunksize], pezzo13[i + c * chunksize],
                              pezzo14[i + c * chunksize])
        Tor_pose_quat = Quaternion(mean11/(i+1), mean12/(i+1), mean13/(i+1), mean14/(i+1))  #problema: questa media mobile non mi convince
        tor_pose_row = tor_quat * Tor_pose_quat.conjugate  # quaternion product
        tor_pose.loc[i + c * chunksize] = [tor_pose_row[0], tor_pose_row[1], tor_pose_row[2], tor_pose_row[3]]
    #Fare la stessa cosa per device 2 e 3
    plt.ylim([-0.3, 1])
    plt.subplot(2, 1, 1)
    plt.title('Quaternions 1,2,3,4 of device 1')
    plt.plot(pezzo11)
    plt.plot(pezzo12)
    plt.plot(pezzo13)
    plt.plot(pezzo14)
    plt.subplot(2, 1, 2)
    plt.title('Tor_pose (unreferenced)')
    plt.plot(tor_pose)
    plt.pause(0.01)


plt.show()
