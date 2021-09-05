# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

data_1 = pd.read_csv('ste13.csv')
data_2 = pd.read_csv('paola11.csv')
data_3 = pd.read_csv('rina10.csv')
data_4 = pd.read_csv('fabio24.csv')
data_5 = pd.read_csv('gabri.csv')
name = 'duty_median_Abdomen'
data_fr_thor = data_1[name]
data_fr_thor = pd.concat([data_fr_thor, data_2[name], data_3[name], data_4[name], data_5[name]],ignore_index=True, axis=1)
print(data_fr_thor)
data_fr_thor.to_csv('frthoraxpopulation.csv', index=False)
