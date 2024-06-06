import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ek1 = pd.read_csv("e-k.1.csv")
ek2 = pd.read_csv("e-k.2.csv")

k = np.zeros(ek1.shape[0])

for i in range(ek1.shape[0]):
    k[i] = 0.5*(ek1["k"][i] + ek2["k"][i])

plt.plot(ek1["e"], ek1["k"], "b+--", alpha=0.5)
plt.plot(ek2["e"], ek2["k"], "b+--", alpha=0.5)
plt.plot(ek1["e"], k, "b-", label="k", linewidth=1)
plt.grid(True)
plt.xlabel("E of forcing")
plt.ylabel("turbulence k")
plt.legend()
plt.savefig("e-k-ueq.jpg")