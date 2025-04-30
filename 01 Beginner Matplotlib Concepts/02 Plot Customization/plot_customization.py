import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Plot Customization]
# Learn how to customize Matplotlib plots for clarity in ML visualization.
# Covers titles, labels, legends, colors, markers, and figure settings.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Setting Titles, Labels, and Legends]
# Customize a scatter plot with titles and labels.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150)
    })
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], color='blue', alpha=0.5, label='Iris Data')
plt.title('Sepal Length vs. Width', fontsize=14, pad=10)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('customized_scatter.png')
plt.close()
print("\nCustomized scatter plot saved as 'customized_scatter.png'")

# %% [3. Customizing Colors, Markers, and Line Styles]
# Customize a line plot.
np.random.seed(42)
days = np.arange(1, 31)
sales = 1000 + np.random.normal(0, 50, 30).cumsum()
plt.plot(days, sales, color='red', linestyle='--', marker='o', markersize=5, label='Sales Trend')
plt.title('Customized Sales Trend', fontsize=14)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend(loc='best')
plt.savefig('customized_line.png')
plt.close()
print("Customized line plot saved as 'customized_line.png'")

# %% [4. Adjusting Figure Size and Resolution]
# Create a large figure with high resolution.
plt.figure(figsize=(10, 6), dpi=100)
plt.hist(df['sepal length (cm)'], bins=20, color='green', alpha=0.7, label='Sepal Length')
plt.title('Sepal Length Distribution', fontsize=14)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.savefig('large_histogram.png')
plt.close()
print("Large histogram saved as 'large_histogram.png'")

# %% [5. Practical ML Application]
# Customize a plot for ML feature comparison.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100)
})
plt.figure(figsize=(8, 5))
plt.scatter(ml_data['feature1'], ml_data['feature2'], color='purple', marker='^', alpha=0.6, label='Features')
plt.title('ML Feature Comparison', fontsize=14)
plt.xlabel('Feature1', fontsize=12)
plt.ylabel('Feature2', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':')
plt.savefig('ml_feature_scatter.png')
plt.close()
print("ML feature scatter saved as 'ml_feature_scatter.png'")

# %% [6. Interview Scenario: Customization]
# Discuss plot customization for ML.
print("\nInterview Scenario: Customization")
print("Q: How would you customize a Matplotlib plot for an ML report?")
print("A: Set title, labels, legend with plt.title, plt.xlabel, plt.legend; adjust colors and figure size.")
print("Key: Clear labels and styles enhance interpretability.")
print("Example: plt.title('Feature Plot', fontsize=14); plt.figure(figsize=(8, 5))")