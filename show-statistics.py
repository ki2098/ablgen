import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv("data/statistics.csv")

plt.plot(d["t"], d["k"], label="k")
plt.plot(d["t"], d["i"], label="i")
plt.plot(d["t"], d["kAvg"], label="kAvg")
plt.plot(d["t"], d["iAvg"], label="iAvg")
plt.grid(True)
plt.ylim([0, 0.1])
plt.yticks(np.arange(0, 0.11, 0.1/10))
plt.legend()
plt.savefig("ffforcing-statistics-feps=1.5e-2.png")