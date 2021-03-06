globals().clear()
# PARAMETERS SELECTION
filename = 'paola11.txt'
#A:sit.wo.su, B:sit, C:supine, D:prone, E:lyingL, F:lyingR, G:standing, I:stairs, L:walkS, M:walkF, N:run, O:cyclette
window_size = 600  # samples inside the window (Must be >=SgolayWindowPCA). 1 minute = 600 samples
SgolayWindowPCA = 31  # original: 31.  MUST BE AN ODD NUMBER
start = 0  # number of initial samples to skip (samples PER device)
stop = 0  # number of sample at which program execution will stop, 0 will run the txt file to the end
incr = 300  # Overlapping between a window and the following. 1=max overlap. MUST BE >= SgolayWindowPCA. The higher the faster
# PLOTTING & COMPUTING OPTIONS
w1plot = 1  # 1 enables plotting quaternions and PCA, 0 disables it
w2plot = 1  # 1 enables plotting respiratory signals and spectrum, 0 disables it
resp_param_plot = 1  # 1 enables plotting respiratory frequency and duty cycle, 0 disables it
batteryplot = 1  # 1 enables plotting battery voltages, 0 disables it
prediction_enabled = 0  # 1 enables posture prediction, 0 disables it
# THRESHOLDS
f_threshold_max = 0.75
f_threshold_min = 0.05


import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics
from pyquaternion import Quaternion
import numpy as np
import scipy.signal
from sklearn.decomposition import PCA
import scipy.stats as stats
import warnings
import time

if prediction_enabled:
    from keras.models import load_model
    test_model = load_model(r'..\Analisi del segnale\Classificatore\complete_GRU.h5')
    labels = ['cyclette', 'stairs', 'supine', 'prone', 'lying_left',
             'lying_right', 'running', 'standing', 'sitting', 'walking', 'unknown']

warnings.simplefilter(action='ignore', category=FutureWarning)

start_time = time.time()

def plotupdate():
    if w1plot:
        # CREAZIONE FINESTRA 1: QUATERNIONI E SEGNALE FILTRATO +PCA
        try:
            plt.figure(1)
            plt.clf()
            plt.subplot(5, 1, 1)
            plt.title('Quaternions of device 1')
            plt.plot(tor['1'], color='red')
            plt.plot(tor['2'], color='green')
            plt.plot(tor['3'], color='skyblue')
            plt.plot(tor['4'], color='orange')
            plt.subplot(5, 1, 2)
            plt.title('Quaternions of device 2')
            plt.plot(abd['1'], color='red')
            plt.plot(abd['2'], color='green')
            plt.plot(abd['3'], color='skyblue')
            plt.plot(abd['4'], color='orange')
            plt.subplot(5, 1, 3)
            plt.title('Quaternions of device 3')
            plt.plot(ref['1'], color='red')
            plt.plot(ref['2'], color='green')
            plt.plot(ref['3'], color='skyblue')
            plt.plot(ref['4'], color='orange')
            plt.subplot(5, 1, 4)
            plt.title('1?? PCA Ab comp + filtering&positive peaks highlighting')
            plt.plot(FuseA_1, color='gold')
            plt.plot(Index_A, EstimSmoothA[Index_A], linestyle='None', marker="*", label='max')
            plt.plot(EstimSmoothA, color='red')
            plt.subplot(5, 1, 5)
            plt.title('1?? PCA Thorax comp + filtering&positive peaks highlighting')
            plt.plot(FuseT_1, color='gold')
            plt.plot(Index_T, EstimSmoothT[Index_T], linestyle='None', marker="*", label='max')
            plt.plot(EstimSmoothT, color='red')
        except Exception as e:
            print("figure 1 error:", e)
    if w2plot:
        # CREAZIONE FINESTRA 2: SEGNALE RESPIRATORIO E SPETTRO
        try:
            plt.figure(2)
            plt.clf()
            plt.subplot(4, 1, 1)
            plt.title('Abdomen signal with (Max-Min) highlight')
            plt.plot(SmoothSmoothA)
            plt.plot(Max_Ind_A, Maxima_A, linestyle='None', marker='+')
            plt.plot(Min_Ind_A, Minima_A, linestyle='None', marker='.')
            plt.subplot(4, 1, 2)
            plt.title('Thorax signal with (Max-Min) highlight')
            plt.plot(SmoothSmoothT)
            plt.plot(Max_Ind_T, Maxima_T, linestyle='None', marker='x')
            plt.plot(Min_Ind_T, Minima_T, linestyle='None', marker='o')
            plt.subplot(4, 1, 3)
            plt.title('Total with (Max-Min) highlight')
            plt.plot(SmoothSmoothTot)
            plt.plot(Max_Ind_Tot, Maxima_Tot, linestyle='None', marker='*')
            plt.plot(Min_Ind_Tot, Minima_Tot, linestyle='None', marker='.')
            plt.subplot(4, 1, 4)
            if index_window > 10:
                plt.plot(f_Tot, pxx_Tot)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Magnitude')
                plt.plot(f_Tot[fBI_Tot + start_Tot], fBmax_Tot, marker='*')
                plt.title('Total Spectrum and maximum')
        except Exception as e:
            print("figure 2 error:", e)
    if resp_param_plot:
    # CREAZIONE FINESTRA 3: FREQUENZA RESPIRATORIA
        try:
            plt.figure(3)
            plt.clf()
            plt.subplot(4, 1, 1)
            plt.plot(indexes, [PCA_T[x][0] for x in range(len(indexes))], color='blue', marker='o')
            plt.ylabel('Resp/min')
            plt.title('fresp from tor')
            plt.subplot(4, 1, 2)
            plt.plot(indexes, [PCA_A[x][0] for x in range(len(indexes))], color='blue', marker='o')
            plt.ylabel('Resp/min')
            plt.title('fresp from abd')
            plt.subplot(4, 1, 3)
            plt.plot(indexes, [Tot_med[x][0] for x in range(len(indexes))], color='blue', marker='o')
            plt.title('total fresp')
            plt.ylabel('Resp/min')
            plt.subplot(4, 1, 4)
            plt.plot(indexes, [min(PCA_T[x][3], PCA_A[x][3], Tot_med[x][3]) for x in range(len(indexes))], color='red', marker='o')
            plt.title('min duty cycle')
        except Exception as e:
            print("figure 3 plotting error:", e)
    if batteryplot:
        # CREAZIONE FINESTRA 4: TENSIONE BATTERIE
        plt.figure(4)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.title('Battery voltage of device 1')
        plt.plot(tor['B'].rolling(window=5).sum()/5 * 1881 / 69280, color='red')
        plt.ylim(top=2.65, bottom=1.5)
        plt.subplot(3, 1, 2)
        plt.title('Battery voltage of device 2')
        plt.plot(abd['B'].rolling(window=5).sum()/5 * 1881 / 69280, color='red')
        plt.ylim(top=2.65, bottom=1.5)
        plt.subplot(3, 1, 3)
        plt.title('Battery voltage of device 3')
        plt.plot(ref['B'].rolling(window=5).sum()/5 * 1881 / 69280, color='red')
        plt.ylim(top=2.65, bottom=1.5)
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

fdev = 10  #frequenza di sampling
print("fdev:", round(fdev, 2), "Hz")
# GLOBAL VARIABLES INITIALIZATION
tor_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
ref_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
abd_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
t1 = pd.DataFrame(columns=['1', '2', '3', '4'])  # segnale toracico da filtrare
a1 = pd.DataFrame(columns=['1', '2', '3', '4'])
pca = PCA(n_components=1)
tor_quat, Tor_pose_quat = Quaternion(), Quaternion()   #quaternioni
ref_quat, Ref_pose_quat = Quaternion(), Quaternion()
abd_quat, Abd_pose_quat = Quaternion(), Quaternion()
FuseT_1, FuseA_1 = [], []
tor_array, abd_array, ref_array = [], [], []
SmoothSmoothA, Max_Ind_A, Maxima_A, Min_Ind_A, Minima_A = [], [], [], [], []
SmoothSmoothT, Max_Ind_T, Maxima_T, Min_Ind_T, Minima_T = [], [], [], [], []
SmoothSmoothTot, Max_Ind_Tot, Maxima_Tot = [], [], []
pxx_Tot, fBI_Tot, start_Tot, fBmax_Tot = 0, 0, 0, 0
f_Tot = [0]
PCA_A, SD_A, PCA_T, SD_T, Tot_med, Tot_Iqr, indexes, predictions = [], [], [], [], [], [], [], []  #per plot parametri respiratori
Min_Ind_Tot, Minima_Tot = [], []  #for plotting

length = len(data)
ncycles = length//3 if stop == 0 else stop
print("Il dataset ha", length, "campioni")
print("Skipping first", start, "data points")
print("Stopping analysis at sample", ncycles)

data = data.drop(['nthvalue'], axis=1)
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
tor = tor.drop(['C'], axis=1)  #il byte che identifica il data loss non serve pi??
abd = abd.drop(['C'], axis=1)
ref = ref.drop(['C'], axis=1)
tor = tor[start:]
abd = abd[start:]
ref = ref[start:]
tor.fillna(method='bfill', inplace=True)
tor = tor.reset_index(drop=True)
abd.fillna(method='bfill', inplace=True)
abd = abd.reset_index(drop=True)
ref.fillna(method='bfill', inplace=True)
ref = ref.reset_index(drop=True)
index_tor = len(tor)
index_abd = len(abd)
index_ref = len(ref)
#  PARTE ITERATIVA DEL CODICE
first_iteration = 1
index_window = 0

# mean of thorax quat in window
tor_pose_w = [statistics.mean(tor.iloc[:, 1]),
              statistics.mean(tor.iloc[:, 2]),
              statistics.mean(tor.iloc[:, 3]),
              statistics.mean(tor.iloc[:, 4])]
abd_pose_w = [statistics.mean(abd.iloc[:, 1]),
              statistics.mean(abd.iloc[:, 2]),
              statistics.mean(abd.iloc[:, 3]),
              statistics.mean(abd.iloc[:, 4])]
ref_pose_w = [statistics.mean(ref.iloc[:, 1]),
              statistics.mean(ref.iloc[:, 2]),
              statistics.mean(ref.iloc[:, 3]),
              statistics.mean(ref.iloc[:, 4])]
Tor_pose, Ref_pose, Abd_pose = [], [], []
while len(Tor_pose) <= max(len(tor), len(abd), len(ref)):
        Tor_pose.append(tor_pose_w)
        Ref_pose.append(ref_pose_w)
        Abd_pose.append(abd_pose_w)
# takes the 4 quaternions, excludes battery voltage and timestamps
tor_array = tor.iloc[:, 1:5].rename_axis().values
ref_array = ref.iloc[:, 1:5].rename_axis().values
abd_array = abd.iloc[:, 1:5].rename_axis().values

while index_window < ncycles-window_size-start:
    print("index_window", index_window)
    #CLASSIFICATION
    if prediction_enabled:
        input = pd.DataFrame(ref_array[index_window:index_window+window_size])  #classifica su finestra
        N_TIME_STEPS = 200
        N_FEATURES = 4
        step = 20
        segments = []
        try:
            #print("len input", len(input))
            for i in range(0, len(input) - N_TIME_STEPS, step):
                quat_1 = input[0].values[i: i + N_TIME_STEPS]
                quat_2 = input[1].values[i: i + N_TIME_STEPS]
                quat_3 = input[2].values[i: i + N_TIME_STEPS]
                quat_4 = input[3].values[i: i + N_TIME_STEPS]
                segments.append([quat_1, quat_2, quat_3, quat_4])
            X = np.asarray(segments).reshape(-1, N_TIME_STEPS, N_FEATURES)
            y_pred = test_model.predict(X, steps=1, verbose=0)
            rounded_y_pred = np.argmax(y_pred, axis=-1)
            #print("raw (last element):", rounded_y_pred[-1])
            print('Prediction:', labels[rounded_y_pred[-1]])
            predictions.append(labels[rounded_y_pred[-1]])
        except Exception as e:
            print("Prediction error:", e)
            predictions.append(labels[-1])

    for i in range(index_window, index_window + window_size):  # campione per campione DENTRO finestra
        # THORAX QUATERNION COMPUTATION
        tor_quat = Quaternion(tor_array[i])  #thorax quat wrt Earth
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
        Ref_pose_quat = Quaternion(Ref_pose[i])  # for quaternion conjugate
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
    # print(t1.isnull().values.any())
    newT = pca.fit_transform(t1.loc[index_window:index_window + window_size])  # PCA thorax. len(newT)= window_size!!
    newA = pca.fit_transform(a1.loc[index_window:index_window + window_size])  # PCA abdomen
    #print("len newt and newa", len(newT), len(newA))
    if first_iteration: #prima iterazione
        FuseT_1 = newT  # serve dalla seconda iterazione in poi
        FuseA_1 = newA
        EstimSmoothT = scipy.signal.savgol_filter(np.ravel(newT), SgolayWindowPCA, 3)  # filtra il segnale
        EstimSmoothA = scipy.signal.savgol_filter(np.ravel(newA), SgolayWindowPCA, 3)
    else:
        FuseT_1 = np.append(FuseT_1, newT[window_size - incr:])  # adds last elements of the computed PCA array
        FuseA_1 = np.append(FuseA_1, newA[window_size - incr:])
        EstimSmoothT = np.append(EstimSmoothT, scipy.signal.savgol_filter(np.ravel(newT[window_size - incr:]), SgolayWindowPCA, 3))  # filtra il segnale
        EstimSmoothA = np.append(EstimSmoothA, scipy.signal.savgol_filter(np.ravel(newA[window_size - incr:]), SgolayWindowPCA, 3))
    #print("len first pc tor", len(EstimSmoothT), "\nlen first pc abd", len(EstimSmoothA))
    try:
        # PEAK DETECTION
        # Thorax
        diff_T = max(EstimSmoothT[index_window:index_window + window_size]) - min(EstimSmoothT[index_window:index_window + window_size])
        thr_T = diff_T * 5 / 100
        Index_T = scipy.signal.find_peaks(EstimSmoothT, distance=6, prominence=thr_T)
        Index_T = Index_T[0]  # ???peak_heights??? selection
        fStimVec_T = []
        if len(Index_T) > 2:  # at least 3 peaks are needed to compute the intrapeak distance
            for i in range(len(Index_T) - 1):
                intrapeak = (Index_T[i + 1] - Index_T[i]) / fdev
                fstim = 1 / intrapeak
                fStimVec_T.append(fstim)
            fStimMean_T = statistics.mean(fStimVec_T)
            fStimstd_T = statistics.stdev(fStimVec_T)
            lowThreshold_T = max(f_threshold_min, (fStimMean_T - fStimstd_T))  # creation of the thorax threshold
            #print("lowThreshold_T", lowThreshold_T)
            f_T, pxx_T = scipy.signal.welch(np.ravel(FuseT_1), window='hamming', fs=10, nperseg=window_size, noverlap=incr,
                                            nfft=window_size,
                                            detrend=False)  # %PCA_1 thoracic spectrum (fT is the nomralized frequency vector)
    except Exception as e:
        print("tor peak detection error:", e)
    try:
        # Abdomen
        diff_A = max(EstimSmoothA[index_window:index_window + window_size]) - min(EstimSmoothA[index_window:index_window + window_size])
        thr_A = diff_A * 5 / 100
        Index_A = scipy.signal.find_peaks(EstimSmoothA, distance=6, prominence=thr_A)  # find peaks
        Index_A = Index_A[0]
        fStimVec_A = []
        if len(Index_A) > 2:  # at least 3 peaks are needed to compute the intrapeak distance
            # print("len index_A", len(Index_A))
            for i in range(len(Index_A) - 1):
                intrapeak = (Index_A[i + 1] - Index_A[i]) / fdev
                fstim = 1 / intrapeak  # intrapeak distance is used to estimate the frequency
                fStimVec_A.append(fstim)
            fStimMean_A = statistics.mean(fStimVec_A)
            fStimstd_A = statistics.stdev(fStimVec_A)
            lowThreshold_A = max(f_threshold_min, (fStimMean_A - fStimstd_A))  # creation of the abdomen threshold
            #print("lowThreshold_A", lowThreshold_A)
            f_A, pxx_A = scipy.signal.welch(np.ravel(FuseA_1), fs=fdev, window='hamming', nperseg=window_size, noverlap=incr,
                                            nfft=window_size,
                                            detrend=False)  # PCA_1 abdomen spectrum (fA is the nomralized frequency vector).

        lowThreshold = min(lowThreshold_A,
                           lowThreshold_T)  # the low threshold is computed as the minimum between the thoracic and the abdominal one
    except Exception as e:
        print("abd peak detection error:", e)
    try:
        # ABDOMEN MAXIMA AND MINIMA DETECTION
        Signal_A = -FuseA_1
        start_A = np.where(f_A > lowThreshold)[0][0] - 1
        end_A = np.where(f_A > f_threshold_max)[0][0]
        fBmax_A = max(pxx_A[start_A:end_A])  # breathing frequency as the highest peak
        fBI_A = np.where(pxx_A[start_A:end_A] == fBmax_A)[0][0]  # max (breathing frequency) postion
        fBspectrum_A = f_A[fBI_A + start_A]  # value retrieved from position in f_A
        f1 = max(f_threshold_min, fBspectrum_A - 0.4)
        f2 = min(fBspectrum_A + 0.4, f_threshold_max)
        ft_pl = f2
        Wn_pl = ft_pl / (fdev / 2)
        b, a = scipy.signal.butter(1, 0.15,
                                   'lowpass')  # low pass filter (Butterworth) b, a = scipy.signal.butter(1, Wn_pl, 'lowpass')
        lowfilt_A = scipy.signal.filtfilt(b, a, np.ravel(Signal_A))
        ft_ph = f1
        Wn_ph = ft_ph / (fdev / 2)
        b, a = scipy.signal.butter(1, Wn_ph,
                                   'highpass')  # high pass filter (the result is the bandpass filtered version)
        bpfilt_A = scipy.signal.filtfilt(b, a, lowfilt_A)
        # parameters setting on empirical basis
        if fBspectrum_A * 60 < 12:
            perc_A = 15
            distance_A = 35  # min peak distance of 35 frames corresponds to a respiratory rate of 17 resp/min
            SgolayWindow = 15
        elif 12 < fBspectrum_A * 60 < 20:
            perc_A = 8
            distance_A = 20  # min peak distance of 20 frames corresponds to a respiratory rate of 30 resp/min
            SgolayWindow = 11
        else:
            perc_A = 5
            distance_A = 9  # min peak distance of 12 frames corresponds to a frequency rate of 50 resp/min
            SgolayWindow = 9
        SmoothSmoothA = scipy.signal.savgol_filter(bpfilt_A, SgolayWindow, 3)
        diff_SSA = max(SmoothSmoothA[index_window:index_window + window_size]) - min(SmoothSmoothA[index_window:index_window + window_size])
        thr_SSA = diff_SSA * perc_A / 100
        Max_Ind_A = scipy.signal.find_peaks(SmoothSmoothA, distance=distance_A, prominence=thr_SSA)
        Max_Ind_A = Max_Ind_A[0]
        Maxima_A = SmoothSmoothA[Max_Ind_A]
        Min_Ind_A, Minima_A = [], []
        for i in range(len(Max_Ind_A) - 1):
            min_value = min(SmoothSmoothA[Max_Ind_A[i]:Max_Ind_A[i + 1]])
            Minima_A.append(min_value)
            min_index = np.argmin(SmoothSmoothA[Max_Ind_A[i]:Max_Ind_A[i + 1]]) + Max_Ind_A[i]
            Min_Ind_A.append(min_index)
    except Exception as e:
        print("abd max-min detection error:", e)
    try:
        # THORAX MAXIMA AND MINIMA DETECTION
        Signal_T = FuseT_1
        start_T = np.where(f_T > lowThreshold)[0][0] - 1
        end_T = np.where(f_T > f_threshold_max)[0][0]
        fBmax_T = max(pxx_T[start_T:end_T])
        fBI_T = np.where(pxx_T[start_T:end_T] == fBmax_T)[0][0]
        fBspectrum_T = f_T[fBI_T + start_T]
        f1 = max(f_threshold_min, fBspectrum_T - 0.4)
        f2 = min(fBspectrum_T + 0.4, f_threshold_max)
        ft_pl = f2
        Wn_pl = ft_pl / (fdev / 2)
        b, a = scipy.signal.butter(1, 0.15, 'lowpass')
        lowfilt_T = scipy.signal.filtfilt(b, a, np.ravel(Signal_T))
        ft_ph = f1
        Wn_ph = ft_ph / (fdev / 2)
        b, a = scipy.signal.butter(1, Wn_ph,
                                   'highpass')  # high pass filter (the result is the bandpass filtered version)
        bpfilt_T = scipy.signal.filtfilt(b, a, lowfilt_T)
        if fBspectrum_T * 60 < 12:
            perc_T = 15
            distance_T = 35  # min peak distance of 35 frames corresponds to a respiratory rate of 17 resp/min
            SgolayWindow = 15
        elif 12 < fBspectrum_T * 60 < 20:
            perc_T = 8
            distance_T = 20  # min peak distance of 20 frames corresponds to a respiratory rate of 30 resp/min
            SgolayWindow = 11
        else:
            perc_T = 5
            distance_T = 9  # min peak distance of 12 frames corresponds to a frequency rate of 50 resp/min
            SgolayWindow = 9
        SmoothSmoothT = scipy.signal.savgol_filter(bpfilt_T, SgolayWindow, 3)
        diff_SST = max(SmoothSmoothT[index_window:index_window + window_size]) - min(SmoothSmoothT[index_window:index_window + window_size])
        thr_SST = diff_SST * perc_T / 100
        Max_Ind_T = scipy.signal.find_peaks(SmoothSmoothT, distance=distance_T, prominence=thr_SST)
        Max_Ind_T = Max_Ind_T[0]
        Maxima_T = SmoothSmoothT[Max_Ind_T]
        Min_Ind_T, Minima_T = [], []
        for i in range(len(Max_Ind_T) - 1):
            min_value = min(SmoothSmoothT[Max_Ind_T[i]:Max_Ind_T[i + 1]])
            Minima_T.append(min_value)
            min_index = np.argmin(SmoothSmoothT[Max_Ind_T[i]:Max_Ind_T[i + 1]]) + Max_Ind_T[i]
            Min_Ind_T.append(min_index)
    except Exception as e:
        print("tor max-min detection error:", e)
    # RESPIRATORY PARAMETERS ABDOMEN
    T_A, Ti_A, Te_A, fB_A = [], [], [], []
    for i in range(len(Min_Ind_A)):
        te = (Min_Ind_A[i] - Max_Ind_A[i]) / fdev
        ti = (Max_Ind_A[i + 1] - Min_Ind_A[i]) / fdev
        Ti_A.append(ti)
        Te_A.append(te)
        ti_te = ti / te
        ttot = ti + te
        fb = 1 / ttot * 60
        T_A.append(ttot)
        fB_A.append(fb)
    try:
        Tmedian_A = statistics.median(T_A)
        Timedian_A = statistics.median(Ti_A)
        Temedian_A = statistics.median(Te_A)
        fBmedian_A = statistics.median(fB_A)
        fBspectrum_A = fBspectrum_A * 60
        duty_median_A = statistics.median([float(Ti_A / T_A) for Ti_A, T_A in zip(Ti_A, T_A)])
        Tstd_A = statistics.stdev(T_A)
        Tistd_A = statistics.stdev(Ti_A)
        Testd_A = statistics.stdev(Te_A)
        fBstd_A = statistics.stdev(fB_A)
        duty_std_A = statistics.stdev([float(Ti_A / T_A) for Ti_A, T_A in zip(Ti_A, T_A)])
        PCA_A.append([fBmedian_A, Timedian_A, Temedian_A, duty_median_A])
        indexes.append(index_window)
        SD_A.append([fBstd_A, Tistd_A, Testd_A, duty_std_A])
        print("fBmedian_Abdomen, Ti_median_Abdomen, Te_median_Abdomen, duty_median_Abdomen\n", [round(i, 2) for i in PCA_A[-1]])
        print("fBirq_Abdomen, Tiirq_Abdomen, Teirq_Abdomen, duty_irq_Abdomen\n", [round(i, 2) for i in SD_A[-1]], "\n")
    except Exception as e:
        print("Errore calcolo parametri abd:", e)
        SD_A.append([np.nan, np.nan, np.nan, np.nan])
        PCA_A.append([np.nan, np.nan, np.nan, np.nan])
        indexes.append(index_window)
    # RESPIRATORY PARAMETERS THORAX
    T_T, Ti_T, Te_T, fB_T = [], [], [], []
    for i in range(len(Min_Ind_T)):
        te = (Min_Ind_T[i] - Max_Ind_T[i]) / fdev
        ti = (Max_Ind_T[i + 1] - Min_Ind_T[i]) / fdev
        Ti_T.append(ti)
        Te_T.append(te)
        ttot = ti + te
        fb = 1 / ttot * 60
        T_T.append(ttot)
        fB_T.append(fb)
    try:
        Tmedian_T = statistics.median(T_T)
        Timedian_T = statistics.median(Ti_T)
        Temedian_T = statistics.median(Te_T)
        fBmedian_T = statistics.median(fB_T)
        fBspectrum_T = fBspectrum_T * 60
        duty_median_T = statistics.median([float(Ti_T / T_T) for Ti_T, T_T in zip(Ti_T, T_T)])
        Tstd_T = stats.iqr(T_T)
        Tistd_T = stats.iqr(Ti_T)
        Testd_T = stats.iqr(Te_T)
        fBstd_T = stats.iqr(fB_T)
        duty_std_T = stats.iqr([float(Ti_T / T_T) for Ti_T, T_T in zip(Ti_T, T_T)])
        PCA_T.append([fBmedian_T, Timedian_T, Temedian_T, duty_median_T])
        SD_T.append([fBstd_T, Tistd_T, Testd_T, duty_std_T])
        print("fBmedian_Thorax, Ti_median_Thorax, Te_median_Thorax, duty_median_Thorax\n", [round(i, 2) for i in PCA_T[-1]])
        print("fBirq_Thorax, Tiirq_Thorax, Teirq_Thorax, duty_irq_Thorax\n", [round(i, 2) for i in SD_T[-1]], "\n")
    except Exception as e:
        print("Errore calcolo parametri tor:", e)
        SD_T.append([np.nan, np.nan, np.nan, np.nan])
        PCA_T.append([np.nan, np.nan, np.nan, np.nan])
    try:
        # TOTAL RESPIRATORY SIGNAL
        SmoothSmoothTot = SmoothSmoothT + SmoothSmoothA
        f_Tot, pxx_Tot = scipy.signal.welch(SmoothSmoothTot, window='hamming', fs=fdev, nperseg=window_size, noverlap=incr,
                                            nfft=window_size, detrend=False)  # Power spectral density computation
        start_Tot = np.where(f_Tot > lowThreshold)[0][0] - 1
        end_Tot = np.where(f_Tot > f_threshold_max)[0][0]
        fBmax_Tot = max(pxx_Tot[start_Tot:end_Tot])
        fBI_Tot = np.where(pxx_Tot[start_Tot:end_Tot] == fBmax_Tot)[0][0]
        nmax = np.where(f_Tot > 1)[0][0]
        fBspectrum_Tot = f_Tot[fBI_Tot + start_Tot]
        if fBspectrum_Tot * 60 < 12:
            perc_Tot = 15
            distance_Tot = 35  # min peak distance of 35 frames corresponds to a respiratory rate of 17 resp/min
            SgolayWindow = 15
        elif 12 < fBspectrum_Tot * 60 < 20:
            perc_Tot = 8
            distance_Tot = 20  # min peak distance of 20 frames corresponds to a respiratory rate of 30 resp/min
            SgolayWindow = 11
        else:
            perc_Tot = 5
            distance_Tot = 9  # min peak distance of 12 frames corresponds to a frequency rate of 50 resp/min
            SgolayWindow = 9
        diff_SSTot = max(SmoothSmoothTot[index_window:index_window + window_size]) - min(SmoothSmoothTot[index_window:index_window + window_size])
        thr_SSTot = diff_SSTot * perc_Tot / 100

        Max_Ind_Tot = scipy.signal.find_peaks(SmoothSmoothTot, distance=distance_Tot, prominence=thr_SSTot)
        Max_Ind_Tot = Max_Ind_Tot[0]
        Maxima_Tot = SmoothSmoothTot[Max_Ind_Tot]
        Min_Ind_Tot, Minima_Tot = [], []
    except Exception as e:
        print("total resp signal error:", e)
    for i in range(len(Max_Ind_Tot) - 1):
        min_value = min(SmoothSmoothTot[Max_Ind_Tot[i]:Max_Ind_Tot[i + 1]])
        Minima_Tot.append(min_value)
        min_index = np.argmin(SmoothSmoothTot[Max_Ind_Tot[i]:Max_Ind_Tot[i + 1]]) + Max_Ind_Tot[i]
        Min_Ind_Tot.append(min_index)
    # TOTAL RESPIRATORY PARAMS
    T_Tot, Ti_Tot, Te_Tot, fB_Tot = [], [], [], []
    for i in range(len(Min_Ind_Tot)):
        te = (Min_Ind_Tot[i] - Max_Ind_Tot[i]) / fdev
        ti = (Max_Ind_Tot[i + 1] - Min_Ind_Tot[i]) / fdev
        Ti_Tot.append(ti)
        Te_Tot.append(te)
        ttot = ti + te
        fb = (1 / ttot) * 60
        T_Tot.append(ttot)
        fB_Tot.append(fb)
    try:
        #MEDIANA
        Tmed_Tot = statistics.median(T_Tot)
        Timed_Tot = statistics.median(Ti_Tot)
        Temed_Tot = statistics.median(Te_Tot)
        fBmed_Tot = statistics.median(fB_Tot)
        fBspectrum_T = fBspectrum_T * 60
        duty_med_Tot = statistics.median([float(Ti_Tot / T_Tot) for Ti_Tot, T_Tot in zip(Ti_Tot, T_Tot)])
        #INTERQUARTILE
        Tirq_Tot = stats.iqr(T_Tot)
        Tiirq_Tot = stats.iqr(Ti_Tot)
        Teirq_Tot = stats.iqr(Te_Tot)
        fBirq_Tot = stats.iqr(fB_Tot)
        duty_irq_Tot = stats.iqr([float(Ti_Tot / T_Tot) for Ti_Tot, T_Tot in zip(Ti_Tot, T_Tot)])
        Tot_med.append([fBmed_Tot, Timed_Tot, Temed_Tot, duty_med_Tot])
        Tot_Iqr.append([fBirq_Tot, Tiirq_Tot, Teirq_Tot, duty_irq_Tot])
        print("fB_median_Tot, Ti_median_Tot, Te_median_Tot, duty_median_Tot\n", [round(i, 2) for i in Tot_med[-1]])
        print("fB_irq_Tot, Ti_irq_Tot, Te_irq_Tot, duty_irq_Tot\n", [round(i, 2) for i in Tot_Iqr[-1]], "\n")
        print("--- Tempo trascorso: %s minuti ---" % round((time.time() - start_time) / 60, 2))
        print("--- Tempo acquisizione:", round(index_window / 600, 2), "minuti ---")
        print("--------------------------------------------------------------------------------------\n")
    except Exception as e:
        print("Errore calcolo parameteri totale:", e)
        Tot_med.append([np.nan, np.nan, np.nan, np.nan])
        Tot_Iqr.append([np.nan, np.nan, np.nan, np.nan])
    first_iteration = 0
    #print("len indexes", len(indexes), "len tot med", len(Tot_med), "len tot PCA_A", len(PCA_A))
    index_window += incr
    #if index_window >= ncycles - index_window:  #mostra il grafico quando ?? quasi alla fine
    #    plotupdate()
    #    plt.pause(0.01)


#END OF WHILE CYCLE. Plot eventually remaining data
try:
    print("final index_window:", index_window)
    print("fBmedian_Abdomen, Ti_median_Abdomen, Te_median_Abdomen, duty_median_Abdomen\n", [round(i, 2) for i in PCA_A[-1]])
    print("fBirq_Abdomen, Tiirq_Abdomen, Teirq_Abdomen, duty_irq_Abdomen\n", [round(i, 2) for i in SD_A[-1]], "\n")
    print("fBmedian_Thorax, Ti_median_Thorax, Te_median_Thorax, duty_median_Thorax\n", [round(i, 2) for i in PCA_T[-1]])
    print("fBirq_Thorax, Tiirq_Thorax, Teirq_Thorax, duty_irq_Thorax\n", [round(i, 2) for i in SD_T[-1]], "\n")
    print("fBmed_Tot, Timed_Tot, Temed_Tot, duty_med_Tot\n", [round(i, 2) for i in Tot_med[-1]])
    print("fBirq_Tot, Tiirq_Tot, Teirq_Tot, duty_irq_Tot\n", [round(i, 2) for i in Tot_Iqr[-1]])
except Exception as e:
    print("Errore print finale", e)

#EXPORT ELABORATED DATA TO CSV FILE

dftot = pd.DataFrame(Tot_med, columns=["fB_median_Tot", "Ti_median_Tot", "Te_median_Tot", "duty_median_Tot"])
dfa = pd.DataFrame(PCA_A, columns=["fBmedian_Abdomen", "Ti_median_Abdomen", "Te_median_Abdomen", "duty_median_Abdomen"])
dft = pd.DataFrame(PCA_T, columns=["fB_median_Thorax", "Ti_median_Thorax", "Te_median_Thorax", "duty_median_Thorax"])
dfiqra = pd.DataFrame(SD_A, columns=["fBirq_Thorax", "Tiirq_Thorax", "Teirq_Thorax", "duty_irq_Thorax"])
dfiqrt = pd.DataFrame(SD_T, columns=["fBirq_Abd", "Tiirq_Abd", "Teirq_Abd", "duty_irq_Abd"])
dfiqrtot = pd.DataFrame(Tot_Iqr, columns=["fBirq_Tot", "Tiirq_Tot", "Teirq_Tot", "duty_irq_Tot"])
tot = pd.concat([dft, dfiqrt, dfa, dfiqra, dftot, dfiqrtot], axis=1)

if prediction_enabled:
    tot['activity'] = predictions

tot['index'] = indexes
tot = tot.set_index('index')
tot.to_csv('total_params_out.csv')
print("--- Computing time: %s minutes ---" % round((time.time() - start_time)/60, 2))
print("END")

plotupdate()
plt.show()

