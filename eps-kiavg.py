import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('data/epsilon-statistics.csv')
plt.plot(d['epsilon'], d['kAvg'], '-gD', label='kAvg')
plt.plot(d['epsilon'], d['iAvg'], '-ro', label='iAvg')
plt.legend()
plt.grid(True)
plt.savefig("epsilon-statistics.png")

