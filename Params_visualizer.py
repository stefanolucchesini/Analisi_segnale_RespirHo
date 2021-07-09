globals().clear()
import matplotlib.pyplot as plt
import pandas as pd
filename = 'total_params_out.csv'
# PLOTTING & COMPUTING OPTIONS
w2plot = 1  # 1 enables plotting respiratory signals and spectrum, 0 disables it
resp_param_plot = 1  # 1 enables plotting respiratory frequency, 0 disables it

data = pd.read_csv(filename, index_col="index")
print(data)
'''
data.columns = ["fB_median_Tot", "Ti_median_Tot", "Te_median_Tot", "duty_median_Tot", "fBmedian_Abdomen",
                "Ti_median_Abdomen", "Te_median_Abdomen", "duty_median_Abdomen", "fB_median_Thorax",
                "Ti_median_Thorax", "Te_median_Thorax", "duty_median_Thorax", "fBirq_Thorax", "Tiirq_Thorax",
                "Teirq_Thorax", "duty_irq_Thorax", "fBirq_Abd", "Tiirq_Abd", "Teirq_Abd", "duty_irq_Abd",
                "fBirq_Tot", "Tiirq_Tot", "Teirq_Tot", "duty_irq_Tot"]
'''
df = data[["duty_median_Abdomen", "duty_median_Thorax", "duty_median_Tot"]]
df.boxplot()
plt.figure(1)
data[["fBmedian_Abdomen", "fB_median_Thorax", "fB_median_Tot"]].plot()
plt.figure(2)
print("END")
plt.show()
