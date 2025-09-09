import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_raw = pd.read_csv('dataset_uf.csv')
df_filtered = pd.read_csv('dataset.csv')

raw = df_raw["token_length"]
filtered = df_filtered["token_length"]

bins = np.logspace(np.log10(raw.min()), np.log10(raw.max()), 30)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# Plot before
axes[0].hist(raw, bins=bins, edgecolor="black")
axes[0].set_xscale("log")
axes[0].set_title("Before filtering")

# Plot after
axes[1].hist(filtered, bins=bins, edgecolor="black")
axes[1].set_xscale("log")
axes[1].set_title("After filtering")

# Common labels
for ax in axes:
    ax.set_xlabel("token_length (log scale)")
    ax.set_ylabel("Number of records")

plt.tight_layout()
plt.show()
