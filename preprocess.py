import pandas as pd
from datetime import timedelta
filename = 'Ale15.txt'
data = pd.read_csv(filename, sep=",|:", header=None, engine='python')
data.columns = ['DevID', 'B', 'C', 'nthvalue', '1', '2', '3', '4', 'day', 'month', 'hour', 'min', 'sec', 'millisec']
data = data.drop(data[data.C == '[FF]'].index)  #toglie tutte le aggiunte dell'interfaccia ANT
data = data.reset_index(drop=True)  # reset the indexes order
data.to_csv('raw.csv')
print(data)
data['DevID'] = data['DevID'].astype(str)
data['DevID'] = data['DevID'].str.replace('[', '')
data['DevID'] = data['DevID'].str.replace(']', '')

tor = data[data['DevID'].str.contains('01')]  # thoracic data
tor.drop(['DevID', 'nthvalue'], axis=1)
tor = tor.reset_index(drop=True)

abd = data[data['DevID'].str.contains('02')]
abd.drop(['DevID', 'nthvalue'], axis=1)
abd = abd.reset_index(drop=True)

ref = data[data['DevID'].str.contains('03')]
ref.drop(['DevID', 'nthvalue'], axis=1)
ref = ref.reset_index(drop=True)


paststamp = timedelta(days=int(tor.loc[0, ['day']]),
                      hours=int(tor.loc[0, ['hour']]),
                      minutes=int(tor.loc[0, ['min']]),
                      seconds=int(tor.loc[0, ['sec']]),
                      milliseconds=int(tor.loc[0, ['millisec']])
                      )
print("lunghezza tor", len(tor))
tor_lost = 0
index = 0
tornew = pd.DataFrame(columns=['B', 'C', '1', '2', '3', '4', 'day', 'month', 'hour', 'min', 'sec', 'millisec'])

empty = {'C': ['FF'], 'B': ['00'], '1': ['00'], '2': ['00'], '3': ['00'], '4': ['00']}
while index < len(tor):
    newstamp = timedelta(days=int(tor.loc[index, ['day']]),
                         hours=int(tor.loc[index, ['hour']]),
                         minutes=int(tor.loc[index, ['min']]),
                         seconds=int(tor.loc[index, ['sec']]),
                         milliseconds=int(tor.loc[index, ['millisec']])
                         )
    diff = newstamp - paststamp
    diffmilli = diff.microseconds/1000   #ATTENZIONE, VA IN OVERFLOW DOPO UN SECONDO!
    diffsec = diff.seconds
    diffmilli += diffsec*1000
    #print("index:", index, "diff", diffmilli)
    if diffmilli > 150: #dato perso
        #print("data loss all'indice", index)
        i = 0
        while diffmilli >= 130:
            i += 1
            diffmilli -= 100
            tor_lost += 1
            #aggiungere campioni persi
            dataloss = pd.DataFrame(empty)
            tornew = tornew.append(dataloss)
    else:  #nessun dato perso
        tornew = tornew.append(tor.iloc[index])
        #print(tornew)
    paststamp = newstamp
    index += 1
tornew = tornew.reset_index(drop=True)
print("campioni persi da tor:", tor_lost)
print("tornew:", tornew)
tornew.to_csv('preprocessed.csv')
print("END")
