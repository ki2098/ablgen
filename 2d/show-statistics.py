import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

d = pd.read_csv("data/statistics.csv")

plt.plot(d["t"], d["k"], label="k")
plt.plot(d["t"], d["kavg"], label="kavg")
plt.grid(True)
# plt.ylim([0, 0.1])
# plt.yticks(np.arange(0, 0.11, 0.1/10))
plt.legend()
plt.savefig("fforcing(e=2,k=0.12).jpg")