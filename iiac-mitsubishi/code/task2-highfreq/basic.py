
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


high_freq = pd.read_pickle('../../data/1 训练用/520-HighFreqDataSet.part1.pkl')
print(high_freq)

plt.plot(high_freq['主轴转速'])
plt.show()
