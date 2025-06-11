import matplotlib.pyplot as plt

# Data grouped by prompt type
data_by_prompt = {
    'simple': [
        (17.52, 0.2752, 'Q4_K_M'),
        (18.72, 0.3444, 'Q8_K_XL'),
        (23.53, 0.3086, 'BF16'),
    ],
    'complex': [
        (26.13, 0.3772, 'Q4_K_M'),
        (28.52, 0.3892, 'Q8_K_XL'),
        (35.12, 0.4051, 'BF16'),
    ]
}

# Define consistent color and marker style per model
model_styles = {
    'Q4_K_M': {'color': 'blue', 'marker': 'o'},
    'Q8_K_XL': {'color': 'green', 'marker': 's'},
    'BF16': {'color': 'red', 'marker': '^'}
}

plt.figure(figsize=(10, 6))

# Plot 'simple' prompt data
simple_data = data_by_prompt['simple']
x_vals, y_vals, labels = zip(*simple_data)
for x, y, model in simple_data:
    style = model_styles[model]
    plt.scatter(x, y, color=style['color'], marker=style['marker'])
    plt.text(x + 0.1, y, model, fontsize=9)
plt.plot(x_vals, y_vals, linestyle='--', linewidth=2, color='black', label='Simple Prompt')

# Plot 'complex' prompt data
complex_data = data_by_prompt['complex']
x_vals, y_vals, labels = zip(*complex_data)
for x, y, model in complex_data:
    style = model_styles[model]
    plt.scatter(x, y, color=style['color'], marker=style['marker'])
    plt.text(x + 0.1, y, model, fontsize=9)
plt.plot(x_vals, y_vals, linestyle=':', linewidth=2, color='gray', label='Complex Prompt')

# Formatting
plt.xlabel("Inference Time per Entry (s)")
plt.ylabel("MCC Score")
plt.title("Inference Time vs MCC Score by Quantized Model")
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig('../gemma_res/speed.png', dpi=400, transparent=True)

plt.show()
