import matplotlib.pyplot as plt
import pandas as pd

ekd = pd.read_csv("e-k.1.csv")

plt.plot(ekd["e"], ekd["k@K=0.01e"], "b+--", label="k@K=0.01e", alpha=0.5, linewidth=1)
plt.plot(ekd["e"], ekd["k@K=0.02e"], "r+--", label="k@K=0.02e", alpha=0.5, linewidth=1)
plt.grid(True)
plt.xlabel("E of forcing")
plt.ylabel("turbulence k")
plt.legend()
plt.savefig("e-k.jpg")