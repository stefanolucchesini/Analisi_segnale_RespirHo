import pandas as pd
import math
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})


data = pd.read_csv('provavar.txt', sep=",|:", header=None, engine='python')
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

data_1 = data_1.iloc[:(max_value) * 256]

for i in range(max_value):
    for j in range(256):
        if data_2['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty data
            data_2 = pd.concat([data_2.loc[:j + i * 256 - 1], empty_row, data_2.loc[j + i * 256:]])
            data_2 = data_2.reset_index(drop=True)

data_2 = data_2.iloc[:(max_value) * 256]

for i in range(max_value):
    for j in range(256):
        if data_3['nthvalue'][j + i * 256] != j:
            empty_row = pd.DataFrame([], index=[j + i * 256])  # creating the empty data
            data_3 = pd.concat([data_3.loc[:j + i * 256 - 1], empty_row, data_3.loc[j + i * 256:]])
            data_3 = data_3.reset_index(drop=True)

data_3 = data_3.iloc[:(max_value) * 256]
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

'''plt.figure(1)
plt.plot(quat1_1, label="quat1_1")
plt.plot(quat1_2, label="quat1_2")
plt.plot(quat1_3, label="quat1_3")
plt.plot(quat1_4, label="quat1_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')'''

'''plt.figure(2)
plt.plot(quat2_1, label="quat2_1")
plt.plot(quat2_2, label="quat2_2")
plt.plot(quat2_3, label="quat2_3")
plt.plot(quat2_4, label="quat2_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')'''

'''plt.figure(3)
plt.plot(quat3_1, label="quat3_1")
plt.plot(quat3_2, label="quat3_2")
plt.plot(quat3_3, label="quat3_3")
plt.plot(quat3_4, label="quat3_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')'''

'''plt.figure(4)
plt.plot(quat1_1, label="quat1_1")
plt.plot(quat2_1, label="quat2_1")
plt.plot(quat3_1, label="quat3_1")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q1')'''

'''plt.figure(5)
plt.plot(quat1_2, label="quat1_2")
plt.plot(quat2_2, label="quat2_2")
plt.plot(quat3_2, label="quat3_2")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q2')'''

'''plt.figure(6)
plt.plot(quat1_3, label="quat1_3")
plt.plot(quat2_3, label="quat2_3")
plt.plot(quat3_3, label="quat3_3")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q3')'''

'''plt.figure(7)
plt.plot(quat1_4, label="quat1_4")
plt.plot(quat2_4, label="quat2_4")
plt.plot(quat3_4, label="quat3_4")
plt.ylim(-1, 1)
plt.legend(loc='upper right')
plt.title('q4')'''

tor = {'quat1_1': quat1_1, 'quat1_2': quat1_2, 'quat1_3': quat1_3, 'quat1_4': quat1_4}
tor = pd.DataFrame(tor)
abd = {'quat2_1': quat2_1, 'quat2_2': quat2_2, 'quat2_3': quat2_3, 'quat2_4': quat2_4}
abd = pd.DataFrame(abd)
ref = {'quat3_1': quat3_1, 'quat3_2': quat3_2, 'quat3_3': quat3_3, 'quat3_4': quat3_4}
ref = pd.DataFrame(ref)

'''plt.figure(8)
plt.subplot(3, 1, 1)
plt.plot(tor)
plt.title('tor')
plt.subplot(3, 1, 2)
plt.plot(abd)
plt.title('abd')
plt.subplot(3, 1, 3)
plt.plot(ref)
plt.title('ref')'''

nan_A = abd['quat2_1'].isna().sum()-1
nan_T = tor['quat1_1'].isna().sum()-1
nan_ref = ref['quat3_1'].isna().sum()-1

dataloss_abd = (nan_A/len(abd))*100
dataloss_tor = (nan_T/len(tor))*100
dataloss_ref = (nan_ref/len(ref))*100

