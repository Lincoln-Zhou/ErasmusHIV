import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


labels = pd.read_csv('balanced.csv')['flag'].to_numpy()
N = len(labels)

predictions = pd.read_csv('predictions.csv')
preds = predictions['prediction'].values.reshape(N, 3)
probs = predictions['prob'].values.reshape(N, 3)

# 1) First prediction
preds_first = preds[:, 0]

# 2) Majority voting
preds_majority = (preds.sum(axis=1) >= 2).astype(int)

# 3) Highest‐probability
idx_max = np.argmax(probs, axis=1)
preds_maxprob = preds[np.arange(N), idx_max]

# 4) No-inconsistent
mask_trusted = (preds.sum(axis=1) == 0) | (preds.sum(axis=1) == 3)
trusted_preds = preds[mask_trusted, 0]    # all three are the same, so pick the first
trusted_labels = labels[mask_trusted]

# Define target names
target_names = ['Exclusion', 'Inclusion']

# — Classification reports
print("=== First‐prediction method ===")
print(classification_report(labels, preds_first, target_names=target_names))
print("=== Majority‐voting method ===")
print(classification_report(labels, preds_majority, target_names=target_names))
print("=== Max‐probability method ===")
print(classification_report(labels, preds_maxprob, target_names=target_names))
print("=== No-inconsistency method ===")
print(classification_report(trusted_labels, trusted_preds, target_names=target_names))

# — Confusion matrices
cms = [
    confusion_matrix(labels, preds_first),
    confusion_matrix(labels, preds_majority),
    confusion_matrix(labels, preds_maxprob),
    confusion_matrix(trusted_labels, trusted_preds)
]
titles = ["First pred", "Majority vote", "Max‐prob", "No-inconsist"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, cm, title in zip(axes, cms, titles):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(ax=ax, colorbar=False)     # use default styling
    ax.set_title(title)

plt.tight_layout()
plt.show()