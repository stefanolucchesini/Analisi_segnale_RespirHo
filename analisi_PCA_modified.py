import pandas as pd
import math
import matplotlib.pyplot as plt
import statistics
from pyquaternion import Quaternion

from sklearn.decomposition import PCA

plt.rcParams.update({'figure.max_open_warning': 0})


data = pd.read_csv('prova3.txt', sep=",|:", header=None, engine='python')
data.columns = ['TxRx', 'DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'None']
# select only the Rx lines
data = data.loc[data['TxRx'] == 'Rx']
data = data.drop(['TxRx', 'None'], axis=1)
data = data.reset_index(drop=True)  # reset the indexes order
# traforming into string in order to remove [ and ] from the file
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
# nth reception and information transmitted
data_1 = data_1[data_1.columns[3:8]]
data_2 = data_2[data_2.columns[3:8]]
data_3 = data_3[data_3.columns[3:8]]

# max value computation to have the number of 256-value blocks
if len(data_3) > len(data_1) & len(data_3) > len(data_1):
    max_value = math.floor(len(data_3) / 256)
elif len(data_1) > len(data_2) & len(data_1) > len(data_3):
    max_value = math.floor(len(data_1) / 256)
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

# add an empty row when there is a "jump" in communication, when the nth value is not received
for i in range(max_value):
    for j in range(256):
        if data_1['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty row
            data_1 = pd.concat([data_1.loc[:j + i * 256 - 1], empty_row, data_1.loc[j + i * 256:]])
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
# quaternion creation, in range [-1;1]
quat1_1 = data_1['1']
for i in range(len(quat1_1)):
    if quat1_1[i] > 127:
        quat1_1[i] = quat1_1[i] - 256
quat1_1 = quat1_1 / 127

quat1_2 = data_1['2']
for i in range(len(quat1_2)):
    if quat1_2[i] > 127:
        quat1_2[i] = quat1_2[i] - 256
quat1_2 = quat1_2 / 127

quat1_3 = data_1['3']
for i in range(len(quat1_3)):
    if quat1_3[i] > 127:
        quat1_3[i] = quat1_3[i] - 256
quat1_3 = quat1_3 / 127

quat1_4 = data_1['4']
for i in range(len(quat1_4)):
    if quat1_4[i] > 127:
        quat1_4[i] = quat1_4[i] - 256
quat1_4 = quat1_4 / 127

quat2_1 = data_2['1']
for i in range(len(quat2_1)):
    if quat2_1[i] > 127:
        quat2_1[i] = quat2_1[i] - 256
quat2_1 = quat2_1 / 127

quat2_2 = data_2['2']
for i in range(len(quat2_2)):
    if quat2_2[i] > 127:
        quat2_2[i] = quat2_2[i] - 256
quat2_2 = quat2_2 / 127

quat2_3 = data_2['3']
for i in range(len(quat2_3)):
    if quat2_3[i] > 127:
        quat2_3[i] = quat2_3[i] - 256
quat2_3 = quat2_3 / 127

quat2_4 = data_2['4']
for i in range(len(quat2_4)):
    if quat2_4[i] > 127:
        quat2_4[i] = quat2_4[i] - 256
quat2_4 = quat2_4 / 127

quat3_1 = data_3['1']
for i in range(len(quat3_1)):
    if quat3_1[i] > 127:
        quat3_1[i] = quat3_1[i] - 256
quat3_1 = quat3_1 / 127

quat3_2 = data_3['2']
for i in range(len(quat3_2)):
    if quat3_2[i] > 127:
        quat3_2[i] = quat3_2[i] - 256
quat3_2 = quat3_2 / 127

quat3_3 = data_3['3']
for i in range(len(quat3_3)):
    if quat3_3[i] > 127:
        quat3_3[i] = quat3_3[i] - 256
quat3_3 = quat3_3 / 127

quat3_4 = data_3['4']
for i in range(len(quat3_4)):
    if quat3_4[i] > 127:
        quat3_4[i] = quat3_4[i] - 256
quat3_4 = quat3_4 / 127

plt.figure(1)
plt.plot(quat1_1, label="quat1_1")
plt.plot(quat1_2, label="quat1_2")
plt.plot(quat1_3, label="quat1_3")
plt.plot(quat1_4, label="quat1_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(quat2_1, label="quat2_1")
plt.plot(quat2_2, label="quat2_2")
plt.plot(quat2_3, label="quat2_3")
plt.plot(quat2_4, label="quat2_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')

plt.figure(3)
plt.plot(quat3_1, label="quat3_1")
plt.plot(quat3_2, label="quat3_2")
plt.plot(quat3_3, label="quat3_3")
plt.plot(quat3_4, label="quat3_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')

plt.figure(4)
plt.plot(quat1_1, label="quat1_1")
plt.plot(quat2_1, label="quat2_1")
plt.plot(quat3_1, label="quat3_1")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q1')

plt.figure(5)
plt.plot(quat1_2, label="quat1_2")
plt.plot(quat2_2, label="quat2_2")
plt.plot(quat3_2, label="quat3_2")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q2')

plt.figure(6)
plt.plot(quat1_3, label="quat1_3")
plt.plot(quat2_3, label="quat2_3")
plt.plot(quat3_3, label="quat3_3")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q3')

plt.figure(7)
plt.plot(quat1_4, label="quat1_4")
plt.plot(quat2_4, label="quat2_4")
plt.plot(quat3_4, label="quat3_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q4')

tor = {'quat1_1': quat1_1, 'quat1_2': quat1_2, 'quat1_3': quat1_3, 'quat1_4': quat1_4}
tor = pd.DataFrame(tor)
abd = {'quat2_1': quat2_1, 'quat2_2': quat2_2, 'quat2_3': quat2_3, 'quat2_4': quat2_4}
abd = pd.DataFrame(abd)
ref = {'quat3_1': quat3_1, 'quat3_2': quat3_2, 'quat3_3': quat3_3, 'quat3_4': quat3_4}
ref = pd.DataFrame(ref)
'''ref = {'quat2_1': quat2_1, 'quat2_2': quat2_2, 'quat2_3': quat2_3, 'quat2_4': quat2_4}
ref = pd.DataFrame(ref)
abd = {'quat3_1': quat3_1, 'quat3_2': quat3_2, 'quat3_3': quat3_3, 'quat3_4': quat3_4}
abd = pd.DataFrame(abd)'''

plt.figure(8)
plt.subplot(3, 1, 1)
plt.plot(tor)
plt.title('tor')
plt.subplot(3, 1, 2)
plt.plot(abd)
plt.title('abd')
plt.subplot(3, 1, 3)
plt.plot(ref)
plt.title('ref')

# Replacing Nan with interpolation
tor.interpolate(method='pchip', inplace=True)
abd.interpolate(method='pchip', inplace=True)
ref.interpolate(method='pchip', inplace=True)
# tor.interpolate(method='spline', order=3, inplace=True) or method='nearest'
# abd.interpolate(method='spline', order=3, inplace=True)
# ref.interpolate(method='spline', order=3, inplace=True)

tor = tor.loc[1:]
tor.fillna(method='bfill', inplace=True)
tor = tor.reset_index(drop=True)
abd = abd.loc[1:]
abd.fillna(method='bfill', inplace=True)
abd = abd.reset_index(drop=True)
ref = ref.loc[1:]
ref.fillna(method='bfill', inplace=True)
ref = ref.reset_index(drop=True)

# pose window selection
plt.figure(9)
plt.subplot(3, 1, 1)
plt.plot(tor)
plt.title('Select pose window')
plt.subplot(3, 1, 2)
plt.plot(abd)
plt.subplot(3, 1, 3)
plt.plot(ref)
'''print('Please click to identify the start')
start = plt.ginput(1)
start = round(start[0][0])
# start = 287 fixed values to compare with file MATLAB
print('Please click to identify the end')
end = plt.ginput(1)
end = round(end[0][0])
# end = 4557'''

start = 600

# interval_tor = (tor.loc[start:end])
interval_tor = (tor.loc[start:])
tor_pose_w = [statistics.mean(interval_tor['quat1_1']), statistics.mean(interval_tor['quat1_2']),
              statistics.mean(interval_tor['quat1_3']), statistics.mean(interval_tor['quat1_4'])]

# interval_abd = (abd.loc[start:end])
interval_abd = (abd.loc[start:])
abd_pose_w = [statistics.mean(interval_abd['quat2_1']), statistics.mean(interval_abd['quat2_2']),
              statistics.mean(interval_abd['quat2_3']), statistics.mean(interval_abd['quat2_4'])]

# interval_ref = (ref.loc[start:end])
interval_ref = (ref.loc[start:])
ref_pose_w = [statistics.mean(interval_ref['quat3_1']), statistics.mean(interval_ref['quat3_2']),
              statistics.mean(interval_ref['quat3_3']), statistics.mean(interval_ref['quat3_4'])]

# Tor_pose = pd.DataFrame(columns=tor.columns)
# for i in range(len(tor)):
#    Tor_pose.loc[i] = tor_pose_w
Tor_pose = []
while len(Tor_pose) < len(tor):
    Tor_pose.append(tor_pose_w)

# Abd_pose = pd.DataFrame(columns=abd.columns)
# for i in range(len(abd)):
#    Abd_pose.loc[i] = abd_pose_w
Abd_pose = []
while len(Abd_pose) < len(abd):
    Abd_pose.append(abd_pose_w)

# Ref_pose = pd.DataFrame(columns=ref.columns)
# for i in range(len(ref)):
#    Ref_pose.loc[i] = ref_pose_w
Ref_pose = []
while len(Ref_pose) < len(ref):
    Ref_pose.append(ref_pose_w)
# array form to create the Quaternion
tor_array = tor.rename_axis().values
# Tor_pose_array = Tor_pose.rename_axis().values
tor_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
for i in range(len(tor)):  # in pyQuaternion the quaternion are only elements 4x1, the product is performed row by row
    tor_quat = Quaternion(tor_array[i])
    Tor_pose_quat = Quaternion(Tor_pose[i])  # quaternion conjugate
    tor_pose_row = tor_quat * Tor_pose_quat.conjugate  # quaternion product
    tor_pose.loc[i] = [tor_pose_row[0], tor_pose_row[1], tor_pose_row[2], tor_pose_row[3]]

abd_array = abd.rename_axis().values
# Abd_pose_array = Abd_pose.rename_axis().values
abd_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
for i in range(len(abd)):
    abd_quat = Quaternion(abd_array[i])
    Abd_pose_quat = Quaternion(Abd_pose[i])
    abd_pose_row = abd_quat * Abd_pose_quat.conjugate
    abd_pose.loc[i] = [abd_pose_row[0], abd_pose_row[1], abd_pose_row[2], abd_pose_row[3]]

ref_array = ref.rename_axis().values
# Ref_pose_array = Ref_pose.rename_axis().values
ref_pose = pd.DataFrame(columns=['1', '2', '3', '4'])
for i in range(len(ref)):
    ref_quat = Quaternion(ref_array[i])
    Ref_pose_quat = Quaternion(Ref_pose[i])
    ref_pose_row = ref_quat * Ref_pose_quat.conjugate
    ref_pose.loc[i] = [ref_pose_row[0], ref_pose_row[1], ref_pose_row[2], ref_pose_row[3]]

'''# selection of the window to be analyzed
plt.figure(10)
plt.subplot(3, 1, 1)
plt.plot(tor)
plt.title('Select window to analyze')
plt.subplot(3, 1, 2)
plt.plot(abd)
plt.subplot(3, 1, 3)
plt.plot(ref)
print('Please click to identify the start')
start_2 = plt.ginput(1)
start_2 = round(start_2[0][0])
# start_2 = 218
print('Please click to identify the end')
end_2 = plt.ginput(1)
end_2 = round(end_2[0][0])
# end_2 = 4665'''

start_2 = 600

'''Tor_Ok = tor_pose.loc[start_2:end_2]
Abd_Ok = abd_pose.loc[start_2:end_2]
Ref_Ok = ref_pose.loc[start_2:end_2]'''

Tor_Ok = tor_pose.loc[start_2:]
Abd_Ok = abd_pose.loc[start_2:]
Ref_Ok = ref_pose.loc[start_2:]

Tor_Ok_array = Tor_Ok.rename_axis().values
Ref_Ok_array = Ref_Ok.rename_axis().values
t1 = pd.DataFrame(columns=['1', '2', '3', '4'])
for i in range(len(Tor_Ok)):
    Tor_Ok_quat = Quaternion(Tor_Ok_array[i])
    Ref_Ok_quat = Quaternion(Ref_Ok_array[i])
    t1_row = Tor_Ok_quat * Ref_Ok_quat.conjugate  # referred to the reference
    t1.loc[i] = [t1_row[0], t1_row[1], t1_row[2], t1_row[3]]

Abd_Ok_array = Abd_Ok.rename_axis().values
Ref_Ok_array = Ref_Ok.rename_axis().values
a1 = pd.DataFrame(columns=['1', '2', '3', '4'])
for i in range(len(Abd_Ok)):
    Abd_Ok_quat = Quaternion(Abd_Ok_array[i])
    Ref_Ok_quat = Quaternion(Ref_Ok_array[i])
    a1_row = Abd_Ok_quat * Ref_Ok_quat.conjugate
    a1.loc[i] = [a1_row[0], a1_row[1], a1_row[2], a1_row[3]]

# moving average
N = 97
interp_T = t1.iloc[:, ].rolling(N, min_periods=49, center=True).mean()
interp_A = a1.iloc[:, ].rolling(N, min_periods=49, center=True).mean()

plt.figure(11)
plt.plot(t1)
plt.title('t1')

plt.figure(12)
plt.plot(a1)
plt.title('a1')

t1 = t1-interp_T
a1 = a1-interp_A
# PCA: the first component is taken for both abdomen and thorax
pca = PCA(n_components=1)  # pca = PCA(n_components=4)
FuseA_1 = pca.fit_transform(a1)  # FuseA_1 = pca.fit_transform(a1)[:,0]  FuseA_2 = pca.fit_transform(a1)[:,1] ...
FuseT_1 = pca.fit_transform(t1)
FuseT_1 = FuseT_1  # problema: rispetto a Matlab prende i segni opposti (direzione opposta), da togliere poi

plt.figure(13)
plt.plot(FuseA_1, label='First PCA Abdomen component')
plt.title('First abdomen PC')
plt.legend(loc='upper right')

plt.figure(14)
plt.plot(FuseT_1, label='First PCA Thorax component')
plt.title('First thorax PC')
plt.legend(loc='upper right')

# TODO check signal sign
