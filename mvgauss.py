import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("mvgauss.csv")
plt.figure(figsize=(5,5))
plt.scatter(df["x"], df["y"], s=1)
plt.show()