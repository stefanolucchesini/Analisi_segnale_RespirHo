globals().clear()
import matplotlib.pyplot as plt
import pandas as pd
filename = 'total_params_out.csv'  #file contenente parametri respiratori calcolati
win = 10  # smooth window in time
data = pd.read_csv(filename)
data['index'] /= 600 #conversione in minuti
#data['index'] /= 60 #conversione in ore
print(data)
index = data['index']
data = data.set_index('index')

dutymedian = data[["duty_median_Tot", "duty_median_Abdomen", "duty_median_Thorax"]]
titetor = data[["Ti_median_Thorax", "Te_median_Thorax"]]

plt.figure(1)
plt.plot(titetor)

plt.figure(2)
plt.subplot(3, 1, 1)
#plt.ylim(14, 20)
plt.title('fB_median_Thorax')
data["fB_median_Thorax"] = data["fB_median_Thorax"].rolling(window=win).sum() / win
data["fB_median_Thorax"].plot()
plt.ylabel('resp/min')
plt.xlabel('time (minutes)')
plt.xlim(0)
plt.subplot(3, 1, 2)
plt.title('fB_median_Abdomen')
data["fBmedian_Abdomen"] = data["fBmedian_Abdomen"].rolling(window=win).sum() / win
data["fBmedian_Abdomen"].plot()
#plt.ylim(14, 20)
plt.xlim(0)
plt.xlabel('time (minutes)')
plt.ylabel('resp/min')
plt.subplot(3, 1, 3)
plt.title('fB_median_Tot')
data["fB_median_Tot"] = data["fB_median_Tot"].rolling(window=win).sum() / win
data["fB_median_Tot"].plot()
plt.xlabel('time (minutes)')
plt.ylabel('resp/min')
#plt.ylim(14, 20)
plt.xlim(0)

plt.figure(3)
plt.title('minimum duty cycle')
listduty = [min(dutymedian.iloc[x, 0], dutymedian.iloc[x, 1], dutymedian.iloc[x, 2]) for x in range(len(index))]
listduty = pd.DataFrame(listduty, index=index)
listduty = listduty.rolling(window=win).sum() / win
plt.plot(listduty)
plt.xlabel('time (minutes)')


print("END")
plt.show()
