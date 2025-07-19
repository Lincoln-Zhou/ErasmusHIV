import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import scienceplots
plt.style.use(['science'])


exp = '1752628641'  # complex
exp = '1752252343'  # simple

# Load label
labels = pd.read_csv('../gemma_res/labels_full.csv')['flag'].to_numpy()
N = len(labels)

# Load predictions and probabilities
predictions = pd.read_csv(f'../gemma_res/experiment_{exp}/predictions.csv')
preds = predictions['prediction'].values.reshape(N, 3)
probs = predictions['prob'].values.reshape(N, 3)

# Load output texts
outputs = pd.read_csv(f'../gemma_res/experiment_{exp}/outputs.csv')['output'].values.reshape(N, 3)

# Tokenize
output_lengths = pd.read_csv(f'../gemma_res/experiment_{exp}/outputs.csv')['length'].values.reshape(N, 3)

# Aggregation Methods

# 1) First prediction
preds_first = preds[:, 0]

# 2) Majority voting
preds_majority = (preds.sum(axis=1) >= 2).astype(int)

# 3) Highest‐probability
idx_max = np.argmax(probs, axis=1)
preds_maxprob = preds[np.arange(N), idx_max]

# 4) No-inconsistent (only trust consistent predictions)
mask_trusted = (preds.sum(axis=1) == 0) | (preds.sum(axis=1) == 3)
trusted_preds = preds[mask_trusted, 0]
trusted_labels = labels[mask_trusted]

# 5) Shortest output
idx_minlen = np.argmin(output_lengths, axis=1)
preds_shortest = preds[np.arange(N), idx_minlen]

# 6) Longest output
idx_maxlen = np.argmax(output_lengths, axis=1)
preds_longest = preds[np.arange(N), idx_maxlen]

# Define target names
target_names = ['Exclusion', 'Inclusion']

# Classification Reports
print("=== First‐prediction method ===")
print(classification_report(labels, preds_first, target_names=target_names, digits=4))
print("=== Majority‐voting method ===")
print(classification_report(labels, preds_majority, target_names=target_names, digits=4))
print("=== Max‐probability method ===")
print(classification_report(labels, preds_maxprob, target_names=target_names, digits=4))
print("=== No-inconsistency method ===")
print(classification_report(trusted_labels, trusted_preds, target_names=target_names, digits=4))
print("=== Shortest-output method ===")
print(classification_report(labels, preds_shortest, target_names=target_names, digits=4))
print("=== Longest-output method ===")
print(classification_report(labels, preds_longest, target_names=target_names, digits=4))

# Confusion Matrices
cms = [
    confusion_matrix(labels, preds_first),
    confusion_matrix(labels, preds_majority),
    confusion_matrix(labels, preds_maxprob),
    confusion_matrix(labels, preds_shortest),
    confusion_matrix(labels, preds_longest),
    confusion_matrix(trusted_labels, trusted_preds),
]
titles = [
    "First Prediction", "Self-consistency", "Max‐probability",
    "Shortest output", "Longest output", "No-inconsist"
]

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
axes = axes.flatten()

for ax, cm, title in zip(axes, cms, titles):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

plt.tight_layout()

plt.savefig(f"../gemma_res/simple_cm.pdf", transparent=True)

plt.show()

# Confusion matrix values
cm = np.array([[788, 169],
               [26, 80]])

# Labels
labels = ['Exclusion', 'Inclusion']

# Plotting
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)

fig, ax = plt.subplots(figsize=(3, 2.5))
disp.plot(cmap='Blues', ax=ax, colorbar=False)

ax.set_xticks([], [])
ax.set_yticks([], [])
plt.title('mBERT')
plt.tight_layout()

plt.savefig(f"../gemma_res/baseline_cm.pdf", transparent=True)
plt.show()
