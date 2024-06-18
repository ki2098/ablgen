import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ek = pd.read_csv("e-k.csv")



plt.plot(ek["e"], ek["k"], "b+-")

plt.grid(True)
plt.xlabel("coefficient of forcing")
plt.ylabel("turbulence k")
plt.savefig("e-k.jpg")