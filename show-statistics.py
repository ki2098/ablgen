import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/statistics.csv")

plt.plot(d["t"], d["k"], label="k")
plt.plot(d["t"], d["i"], label="i")
plt.legend()
plt.show()