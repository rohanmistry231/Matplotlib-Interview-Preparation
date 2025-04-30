import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Subplots and Layouts]
# Learn how to create and manage subplots for ML visualizations.
# Covers creating subplots, adjusting spacing, and sharing axes.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Creating Multiple Plots in a Grid]
# Create a 2x2 subplot grid for Iris features.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
        'petal length (cm)': np.random.normal(3.7, 1.8, 150),
        'petal width (cm)': np.random.normal(1.2, 0.6, 150)
    })

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].hist(df['sepal length (cm)'], bins=20, color='blue', alpha=0.7)
axes[0, 0].set_title('Sepal Length')
axes[0, 1].hist(df['sepal width (cm)'], bins=20, color='green', alpha=0.7)
axes[0, 1].set_title('Sepal Width')
axes[1, 0].hist(df['petal length (cm)'], bins=20, color='red', alpha=0.7)
axes[1, 0].set_title('Petal Length')
axes[1, 1].hist(df['petal width (cm)'], bins=20, color='purple', alpha=0.7)
axes[1, 1].set_title('Petal Width')
plt.suptitle('Iris Feature Distributions', fontsize=16)
plt.savefig('subplot_grid.png')
plt.close()
print("\nSubplot grid saved as 'subplot_grid.png'")

# %% [3. Adjusting Subplot Spacing]
# Create subplots with adjusted spacing.
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], color='blue', alpha=0.5)
axes[0, 0].set_title('Sepal Length vs. Width')
axes[0, 1].scatter(df['petal length (cm)'], df['petal width (cm)'], color='green', alpha=0.5)
axes[0, 1].set_title('Petal Length vs. Width')
axes[1, 0].plot(df['sepal length (cm)'], color='red', label='Sepal Length')
axes[1, 0].set_title('Sepal Length Trend')
axes[1, 1].plot(df['petal length (cm)'], color='purple', label='Petal Length')
axes[1, 1].set_title('Petal Length Trend')
plt.tight_layout()
plt.suptitle('Mixed Subplots', fontsize=16, y=1.05)
plt.savefig('subplot_spacing.png')
plt.close()
print("Subplot with spacing saved as 'subplot_spacing.png'")

# %% [4. Sharing Axes for Consistent Scales]
# Create subplots with shared axes.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].hist(df['sepal length (cm)'], bins=20, color='blue', alpha=0.7)
axes[0].set_title('Sepal Length')
axes[0].set_ylabel('Frequency')
axes[1].hist(df['petal length (cm)'], bins=20, color='red', alpha=0.7)
axes[1].set_title('Petal Length')
plt.suptitle('Shared Y-Axis Histograms', fontsize=16)
plt.savefig('shared_axes.png')
plt.close()
print("Shared axes subplot saved as 'shared_axes.png'")

# %% [5. Practical ML Application]
# Visualize ML features in subplots.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'target': np.random.choice([0, 1], 100)
})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for target in [0, 1]:
    subset = ml_data[ml_data['target'] == target]
    axes[0].hist(subset['feature1'], bins=15, alpha=0.5, label=f'Class {target}')
    axes[1].hist(subset['feature2'], bins=15, alpha=0.5, label=f'Class {target}')
axes[0].set_title('Feature1 by Class')
axes[0].set_ylabel('Frequency')
axes[1].set_title('Feature2 by Class')
axes[0].legend()
axes[1].legend()
plt.suptitle('ML Feature Distributions', fontsize=16)
plt.tight_layout()
plt.savefig('ml_subplots.png')
plt.close()
print("ML subplots saved as 'ml_subplots.png'")

# %% [6. Interview Scenario: Subplots]
# Discuss subplots for ML.
print("\nInterview Scenario: Subplots")
print("Q: How would you create a grid of plots in Matplotlib for ML features?")
print("A: Use plt.subplots to create a grid and plot on each axis with axes[i, j].")
print("Key: Subplots organize multiple visualizations for comparison.")
print("Example: fig, axes = plt.subplots(2, 2); axes[0, 0].hist(df['col'])")