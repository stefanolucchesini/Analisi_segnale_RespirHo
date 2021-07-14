globals().clear()
import matplotlib.pyplot as plt
import pandas as pd
filename = 'totalpar.csv'
# PLOTTING & COMPUTING OPTIONS
w2plot = 1  # 1 enables plotting respiratory signals and spectrum, 0 disables it
resp_param_plot = 1  # 1 enables plotting respiratory frequency, 0 disables it

data = pd.read_csv(filename)
data['index'] /= 600 #conversione in minuti
index = data['index']
data = data.set_index('index')
print(data)
'''
data.columns = ["fB_median_Tot", "Ti_median_Tot", "Te_median_Tot", "duty_median_Tot", "fBmedian_Abdomen",
                "Ti_median_Abdomen", "Te_median_Abdomen", "duty_median_Abdomen", "fB_median_Thorax",
                "Ti_median_Thorax", "Te_median_Thorax", "duty_median_Thorax", "fBirq_Thorax", "Tiirq_Thorax",
                "Teirq_Thorax", "duty_irq_Thorax", "fBirq_Abd", "Tiirq_Abd", "Teirq_Abd", "duty_irq_Abd",
                "fBirq_Tot", "Tiirq_Tot", "Teirq_Tot", "duty_irq_Tot"]
'''
df = data[["duty_median_Abdomen", "duty_median_Thorax", "duty_median_Tot"]]
df2 = data[["duty_median_Tot", "duty_median_Abdomen", "duty_median_Thorax"]]

plt.figure(1)
df.boxplot()

plt.figure(2)
plt.subplot(3, 1, 1)
plt.title('fB_median_Thorax')
data["fB_median_Thorax"].plot()
plt.ylabel('resp/min')
plt.subplot(3, 1, 2)
plt.title('fB_median_Abdomen')
data["fBmedian_Abdomen"].plot()
plt.ylabel('resp/min')
plt.subplot(3, 1, 3)
plt.title('fB_median_Tot')
data[ "fB_median_Tot"].plot()
plt.xlabel('time (minutes)')
plt.ylabel('resp/min')

plt.figure(3)
plt.title('minimum duty cycle')
plt.plot(index, [min(df2.iloc[x, 0], df2.iloc[x, 1], df2.iloc[x, 2]) for x in range(len(index))])

print("END")
plt.show()
