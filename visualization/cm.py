import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Class names
class_names = ["Exclusion", "Inclusion"]

# Folder order and titles
folders = ["4bq_sp", "8bq_sp", "16b_sp", "4bq_cp", "8bq_cp", "16b_cp"]
titles = [
    "4-bit Simple Prompt", "8-bit Simple Prompt", "16-bit Simple Prompt",
    "4-bit Complex Prompt", "8-bit Complex Prompt", "16-bit Complex Prompt"
]

labels = pd.read_csv('../gemma_res/labels.csv')['flag'].to_numpy()

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
axes = axes.flatten()

for i, (folder, title) in enumerate(zip(folders, titles)):
    # Load predictions from .npy
    y_pred = np.load(f"../gemma_res/{folder}/predictions.npy")

    # Compute confusion matrix
    cm = confusion_matrix(labels, y_pred, labels=[0, 1])

    # Plot on corresponding subplot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(title)

# Adjust layout
plt.tight_layout()

plt.savefig("../gemma_res/confusion_matrix.png", dpi=400, transparent=True)
plt.show()
