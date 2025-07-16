# This script is used for data analysis and visualization only, it's a mess and is not meant for production env deployment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu


exp = '1752628641'  # complex
# exp = '1752252343'  # simple


def remove_outlier(data: pd.DataFrame | pd.Series, percentage: float = 0.005):
    lower = data.quantile(percentage)
    upper = data.quantile(1 - percentage)

    return data[(data < upper) & (data > lower)]


# Load data
labels_df = pd.read_csv("../gemma_res/labels_full.csv")  # Contains 'flag' and 'length'
outputs_df = pd.read_csv(f"../gemma_res/experiment_{exp}/outputs.csv")  # Contains 'output'
preds_df = pd.read_csv(f"../gemma_res/experiment_{exp}/predictions.csv")  # Contains 'prediction' and 'prob'

# Constants
N = len(labels_df)

# Reshape to (N, 3)
pred_matrix = preds_df['prediction'].values.reshape(N, 3)
output_matrix = outputs_df['output'].values.reshape(N, 3)

# Majority voting
maj_vote = np.round(np.mean(pred_matrix, axis=1)).astype(int)

# Ground truth and correctness
y_true = labels_df['flag'].values
correctness = (maj_vote == y_true).astype(int)
labels_df['correct'] = correctness

# --- Updated Analysis 1: Input Length vs Correctness (fewer bins) ---
# Use quantile-based binning
labels_df['length_bin'] = pd.qcut(labels_df['length'], q=10, duplicates='drop')

# Convert interval bin labels to readable integer range strings
labels_df['length_bin_label'] = labels_df['length_bin'].apply(
    lambda x: f"{int(x.left) + 1}–{int(x.right)}"
)

# Group and compute accuracy per bin
binned_accuracy = labels_df.groupby('length_bin_label')['correct'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='length_bin_label', y='correct', data=binned_accuracy)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Input Length Bin (Token Count)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Input Length (Quantile Binning with Integer Labels)")
plt.tight_layout()
plt.show()

# -------------------------------
# Analysis 2: Output Length vs Correctness
# -------------------------------
output_lengths = outputs_df['length'].values.reshape(N, 3)

avg_output_length = output_lengths.mean(axis=1)

correct_df = pd.DataFrame({
    "avg_output_length": avg_output_length,
    "correct": correctness
})


# Bin output lengths using quantiles
correct_df['output_length_bin'] = pd.qcut(correct_df['avg_output_length'], q=10, duplicates='drop')
correct_df['output_bin_label'] = correct_df['output_length_bin'].apply(
    lambda x: f"{int(x.left) + 1}–{int(x.right)}"
)

# Group by bin and compute accuracy
binned_accuracy = correct_df.groupby('output_bin_label')['correct'].mean().reset_index()

# Plot bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='output_bin_label', y='correct', data=binned_accuracy)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Average Output Length Bin")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Output Length (Quantile Binning)")
plt.tight_layout()
plt.show()

# --- Updated Analysis 3: Input Length vs Output Length with Log Scale ---
avg_output_length = output_lengths.mean(axis=1)

correlation_df = pd.DataFrame({
    "input_length": labels_df['length'],
    "avg_output_length": avg_output_length
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.log10(correlation_df['input_length']), y=np.log10(correlation_df['avg_output_length']), alpha=0.4)
sns.regplot(
    x=np.log10(correlation_df['input_length']),
    y=np.log10(correlation_df['avg_output_length']),
    scatter=False,
    color='red',
    ci=95
)

plt.xlabel("Input Length (log scale)")
plt.ylabel("Average Output Length (log scale)")
plt.title("Input Length vs Output Length (Log-Log Scale)")
plt.tight_layout()
plt.show()

spearman_corr, spearman_pval = spearmanr(
    np.log10(correlation_df['input_length']),
    np.log10(correlation_df['avg_output_length'])
)

pearson_corr, pearson_pval = pearsonr(
    correlation_df['input_length'],
    correlation_df['avg_output_length']
)

print(f"Pearson correlation: {pearson_corr:.3f} (p = {pearson_pval:.2e})")
print(f"Spearman correlation (log-log): {spearman_corr:.3f} (p = {spearman_pval:.2e})")

df = pd.DataFrame({
    "correct": correctness,
    "input_length": labels_df['length'],
    "avg_output_length": avg_output_length
})

# Split data by correctness
input_correct = remove_outlier(df[df['correct'] == 1]['input_length'])
input_incorrect = remove_outlier(df[df['correct'] == 0]['input_length'])
output_correct = remove_outlier(df[df['correct'] == 1]['avg_output_length'])
output_incorrect = remove_outlier(df[df['correct'] == 0]['avg_output_length'])

# --- Test 1: Input length difference between correct and incorrect ---
tstat_input, pval_input = ttest_ind(input_correct, input_incorrect, equal_var=False, alternative='two-sided')
u_input, pval_u_input = mannwhitneyu(input_correct, input_incorrect, alternative='greater')

# --- Test 2: Output length difference between correct and incorrect ---
tstat_output, pval_output = ttest_ind(output_correct, output_incorrect, equal_var=False, alternative='two-sided')
u_output, pval_u_output = mannwhitneyu(output_correct, output_incorrect, alternative='two-sided')

# Results
res = {
    "Input Length": {
        "T-test p-value": pval_input,
        "Mann-Whitney U p-value": pval_u_input
    },
    "Output Length": {
        "T-test p-value": pval_output,
        "Mann-Whitney U p-value": pval_u_output
    }
}

print(res)
