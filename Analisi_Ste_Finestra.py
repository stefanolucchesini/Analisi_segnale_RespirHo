globals().clear()

window_size = 31  # samples inside the window (must be >=SgolayWindowPCA!!)   97 for original MA
SgolayWindowPCA = 15  # original: 31

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

plt.rcParams.update({'figure.max_open_warning': 0})

data = pd.read_csv('Stefano_L_A_new.txt', sep=",|:", header=None, engine='python')
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
tor_quat = Quaternion()
Tor_pose_quat = Quaternion()
tor_pose = pd.DataFrame(columns=['1', '2', '3', '4'])

ref_quat = Quaternion()
Ref_pose_quat = Quaternion()
ref_pose = pd.DataFrame(columns=['1', '2', '3', '4'])

abd_quat = Quaternion()
Abd_pose_quat = Quaternion()
abd_pose = pd.DataFrame(columns=['1', '2', '3', '4'])

FuseT_1, FuseA_1 = [], []
Tor_pose, Ref_pose, Abd_pose = [], [], []

index_data = 0  # global index for total data
count = 0
index_tor, index_abd, index_ref = 0, 0, 0  # indexes for devices
index_tor_old, index_abd_old, index_ref_old = 0, 0, 0
index_window = 0  # for computing things inside the window
flag = 0  # used for plotting after first window is available
length = len(data)
print("Il dataset ha", length, "campioni")

#  PARTE ITERATIVA DEL CODICE

while index_data < length:
    print("INDEX:", index_data)
    # transforming into string in order to remove [ and ] from the file\
    data.iloc[index_data] = data.iloc[index_data].astype(str)
    data.iloc[index_data] = data.iloc[index_data].str.replace('[', '')
    data.iloc[index_data] = data.iloc[index_data].str.replace(']', '')
    data.iloc[index_data, 1:8] = data.iloc[index_data, 1:8].apply(int,
                                                                  base=16)  # convert to base 10 everything but DevID

    # Mette NAN ai quaternioni se il pacchetto è invalido
    if data.iloc[index_data, 2] == 255:  # 2 è la colonna C
        data.iloc[index_data, 4:8] = np.nan
        data.iloc[index_data, 1] = np.nan  # mette nan anche al valore della batteria
        print("Il nan è a", index_data)

    # Creazione dataframe del Reference (3)
    check = data.iloc[index_data].str.contains('03')
    if check['DevID'] == True:  # se device id è 3
        # mette il dato nel dataframe del terzo device
        ref = ref.append(data.iloc[index_data])
        ref = ref.reset_index(drop=True)
        ref = ref.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        ref = ref.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(3, index_ref)  # device 3 conversion
        index_ref += 1

    # Creazione dataframe dell'addome (2)
    check = data.iloc[index_data].str.contains('2')
    if check['DevID'] == True:  # se device id è 2
        # mette il dato nel dataframe del terzo device
        abd = abd.append(data.iloc[index_data])
        abd = abd.reset_index(drop=True)
        abd = abd.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        abd = abd.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(2, index_abd)  # device 1 conversion
        index_abd += 1

    # Creazione dataframe del torace (1)
    check = data.iloc[index_data].str.contains('01')
    if check['DevID'] == True:  # se device id è 1
        # mette il dato nel dataframe del terzo device
        tor = tor.append(data.iloc[index_data])
        tor = tor.reset_index(drop=True)
        tor = tor.drop(['DevID', 'C', 'nthvalue'], axis=1)  # Leave only battery and quaternions data
        tor = tor.astype(float)
        # conversion of quaternions in range [-1:1]
        quatsconv(1, index_tor)  # device 1 conversion
        index_tor += 1

    # INSIDE THE WINDOW
    if index_tor + index_abd + index_ref >= 3 * window_size + 3:
        print("index_tor", index_tor, "index_abd", index_abd, "index_ref", index_ref)
        if index_tor > index_tor_old and index_abd > index_abd_old and index_ref > index_ref_old:
            flag += 1  # time to plot
            index_tor_old = index_tor
            index_ref_old = index_ref
            index_abd_old = index_abd
            # inizia a lavorare sui dati quando la prima finestra è piena
            # INTERPOLATE NON VA...fa divergere tutto
            # tor.interpolate(method='pchip', inplace=True)
            # abd.interpolate(method='pchip', inplace=True)
            # ref.interpolate(method='pchip', inplace=True)
            # tor = tor.loc[1:]
            tor.iloc[index_window:index_window + window_size] = tor.iloc[
                                                                index_window:index_window + window_size].fillna(
                method='ffill').fillna(method='bfill')
            # tor = tor.reset_index(drop=True)
            # abd = abd.loc[1:]
            abd.iloc[index_window:index_window + window_size] = abd.iloc[
                                                                index_window:index_window + window_size].fillna(
                method='ffill').fillna(method='bfill')
            # abd = abd.reset_index(drop=True)
            # ref = ref.loc[1:]
            ref.iloc[index_window:index_window + window_size] = ref.iloc[
                                                                index_window:index_window + window_size].fillna(
                method='ffill').fillna(method='bfill')
            # ref = ref.reset_index(drop=True)
            print("index_window+window_size:", index_window + window_size)
            # print("tor just after interpol\n", tor.head(index_window+window_size))
            # mean of thorax quat in window
            tor_pose_w = [statistics.mean(tor.iloc[index_window:index_window + window_size, 1]),
                          statistics.mean(tor.iloc[index_window:index_window + window_size, 2]),
                          statistics.mean(tor.iloc[index_window:index_window + window_size, 3]),
                          statistics.mean(tor.iloc[index_window:index_window + window_size, 4])]
            abd_pose_w = [statistics.mean(abd.iloc[index_window:index_window + window_size, 1]),
                          statistics.mean(abd.iloc[index_window:index_window + window_size, 2]),
                          statistics.mean(abd.iloc[index_window:index_window + window_size, 3]),
                          statistics.mean(abd.iloc[index_window:index_window + window_size, 4])]
            ref_pose_w = [statistics.mean(ref.iloc[index_window:index_window + window_size, 1]),
                          statistics.mean(ref.iloc[index_window:index_window + window_size, 2]),
                          statistics.mean(ref.iloc[index_window:index_window + window_size, 3]),
                          statistics.mean(ref.iloc[index_window:index_window + window_size, 4])]
            while len(Tor_pose) < len(tor):
                Tor_pose.append(tor_pose_w)
                Ref_pose.append(ref_pose_w)
                Abd_pose.append(abd_pose_w)
            # takes the 4 quaternions, excludes battery voltage
            tor_array = tor.iloc[:index_window + window_size, 1:5].rename_axis().values
            ref_array = ref.iloc[:index_window + window_size, 1:5].rename_axis().values
            abd_array = abd.iloc[:index_window + window_size, 1:5].rename_axis().values
            # print("ref", ref.head(index_window+window_size))
            Tor_Ok_array = tor_pose.rename_axis().values
            Ref_Ok_array = ref_pose.rename_axis().values
            t1 = pd.DataFrame(columns=['1', '2', '3', '4'])
            a1 = pd.DataFrame(columns=['1', '2', '3', '4'])

            for i in range(index_window, index_window + window_size):  # campione per campione DENTRO finestra
                # THORAX QUATERNION COMPUTATION
                tor_quat = Quaternion(tor_array[i])
                Tor_pose_quat = Quaternion(Tor_pose[i])  # quaternion
                tor_pose_row = tor_quat * Tor_pose_quat.conjugate  # quaternion product
                tor_pose.loc[i] = [tor_pose_row[0], tor_pose_row[1], tor_pose_row[2], tor_pose_row[3]]
                # ABDOMEN QUATERNION COMPUTATION
                abd_quat = Quaternion(abd_array[i])
                Abd_pose_quat = Quaternion(Abd_pose[i])  # quaternion
                abd_pose_row = abd_quat * Abd_pose_quat.conjugate  # quaternion product
                abd_pose.loc[i] = [abd_pose_row[0], abd_pose_row[1], abd_pose_row[2], abd_pose_row[3]]
                # REFERENCE QUATERNION COMPUTATION
                ref_quat = Quaternion(ref_array[i])
                Ref_pose_quat = Quaternion(Ref_pose[i])  # quaternion conjugate
                ref_pose_row = ref_quat * Ref_pose_quat.conjugate  # quaternion product
                ref_pose.loc[i] = [ref_pose_row[0], ref_pose_row[1], ref_pose_row[2], ref_pose_row[3]]
                # THORAX COMPONENT
                Tor_Ok_quat = Quaternion(tor_pose.loc[i].rename_axis().values)
                Ref_Ok_quat = Quaternion(ref_pose.loc[i].rename_axis().values)
                t1_row = Tor_Ok_quat * Ref_Ok_quat.conjugate  # referred to the reference
                t1.loc[i] = [t1_row[0], t1_row[1], t1_row[2], t1_row[3]]
                # ABDOMEN COMPONENT
                Abd_Ok_quat = Quaternion(abd_pose.loc[i].rename_axis().values)
                a1_row = Abd_Ok_quat * Ref_Ok_quat.conjugate  # referred to the reference
                a1.loc[i] = [a1_row[0], a1_row[1], a1_row[2], a1_row[3]]

            # fine del calcolo dentro la finestra
            interp_T = t1.loc[index_window:index_window + window_size].rolling(window_size,
                                                                               min_periods=math.floor(window_size / 2),
                                                                               center=True).mean()
            interp_A = a1.loc[index_window:index_window + window_size].rolling(window_size,
                                                                               min_periods=math.floor(window_size / 2),
                                                                               center=True).mean()
            t1 = t1 - interp_T
            a1 = a1 - interp_A
            print(t1.isnull().values.any())
            newT = pca.fit_transform(t1)  # PCA thorax
            newA = pca.fit_transform(a1)  # PCA abdomen
            if index_window == 0:
                FuseT_1 = newT
                FuseA_1 = newA
            else:
                FuseT_1 = np.append(FuseT_1, newT[-1])  # adds last element of the computed PCA array
                FuseA_1 = np.append(FuseA_1, newA[-1])
            print(len(FuseT_1), len(FuseA_1))
            # PEAK DETECTION
            EstimSmoothT = scipy.signal.savgol_filter(np.ravel(FuseT_1), SgolayWindowPCA, 3)  # filtra il segnale
            diff_T = max(EstimSmoothT) - min(EstimSmoothT)
            thr_T = diff_T * 5 / 100
            Index_T = scipy.signal.find_peaks(EstimSmoothT, distance=6, prominence=thr_T)
            Index_T = Index_T[0]
            EstimSmoothA = scipy.signal.savgol_filter(np.ravel(FuseA_1), SgolayWindowPCA,
                                                      3)  # Savitzky-Golay filter application
            diff_A = max(EstimSmoothA) - min(EstimSmoothA)
            thr_A = diff_A * 5 / 100
            Index_A = scipy.signal.find_peaks(EstimSmoothA, distance=6, prominence=thr_A)  # find peaks
            Index_A = Index_A[0]

            index_window += 1

    index_data += 1  # global
    if flag >= 1:
        flag = 0
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
        plt.plot(tor[['1', '2', '3', '4']])
        plt.subplot(3, 1, 2)
        # plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
        # plt.plot(abd[['1', '2', '3', '4']])
        plt.title('1° PCA Ab comp + filtering&positive peaks highlighting')
        plt.plot(FuseA_1, color='gold')
        plt.plot(Index_A, EstimSmoothA[Index_A], linestyle='None', marker="*", label='max')
        plt.plot(EstimSmoothA, color='red')
        # plt.subplot(3, 1, 2)
        # plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
        # plt.plot(ref[['1', '2', '3', '4']])
        plt.subplot(3, 1, 3)
        plt.title('1° PCA Th comp + filtering&positive peaks highlighting')
        plt.plot(FuseT_1, color='gold')
        plt.plot(Index_T, EstimSmoothT[Index_T], linestyle='None', marker="*", label='max')
        plt.plot(EstimSmoothT, color='red')

        plt.pause(0.01)

# plot eventually remaining data
plt.clf()
plt.subplot(3, 1, 1)
plt.title('Quaternions 1,2,3,4 of device 1 (thorax)')
plt.plot(tor[['1', '2', '3', '4']])
# plt.subplot(4, 1, 2)
# plt.title('Quaternions 1,2,3,4 of device 2 (abdomen)')
# plt.plot(abd[['1', '2', '3', '4']])
plt.subplot(3, 1, 2)
plt.title('1° PCA Ab comp + filtering&positive peaks highlighting')
plt.plot(FuseA_1, color='gold')
plt.plot(Index_A, EstimSmoothA[Index_A], linestyle='None', marker="*", label='max')
plt.plot(EstimSmoothA, color='red')
# plt.title('Quaternions 1,2,3,4 of device 3 (reference)')
# plt.plot(ref[['1', '2', '3', '4']])
plt.subplot(3, 1, 3)
plt.title('1° PCA Thorax comp + filtering&positive peaks highlighting')
plt.plot(FuseT_1, color='gold')
plt.plot(Index_T, EstimSmoothT[Index_T], linestyle='None', marker="*", label='max')
plt.plot(EstimSmoothT, color='red')

plt.show()

# data1.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_1after.csv', index=False, header=True)
# data2.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_2after.csv', index=False, header=True)
# data3.to_csv(r'C:\Users\Stefano\Desktop\Analisi del segnale\data_3after.csv', index=False, header=True)
