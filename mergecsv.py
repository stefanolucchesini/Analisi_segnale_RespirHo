import pandas as pd
import numpy as np

filename1 = 'total_params_out1.csv'  #prime 3 ore
filename2 = 'total_params_out2.csv'  #seconde 3 ore
#filename3 = 'total_params_out3.csv'  #ultime 3 ore

blocco1 = pd.read_csv(filename1, index_col="index")
last_index1 = blocco1.tail(1).index.item()+300
print("Ultimo indice del primo blocco:", last_index1-300, "\nCorrispondente a", last_index1/600, "minuti")
blocco2 = pd.read_csv(filename2)
#blocco3 = pd.read_csv(filename3, index_col="index")
blocco2 = blocco2[4:]
blocco2['index'] -= 1200
blocco2['index'] += last_index1
blocco2 = blocco2.set_index('index')
result = pd.concat([blocco1, blocco2])
#print(blocco1)
print(blocco2)
#print(result)
result.to_csv('totalpar.csv')
