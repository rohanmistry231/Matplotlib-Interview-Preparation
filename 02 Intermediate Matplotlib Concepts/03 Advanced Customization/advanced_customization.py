import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Advanced Customization]
# Learn advanced Matplotlib customization for ML visualizations.
# Covers annotations, colormaps, and axis customization.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Annotating Plots]
# Annotate key points in a scatter plot.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150)
    })
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], color='blue', alpha=0.5)
max_point = df.iloc[df['sepal length (cm)'].idxmax()]
plt.annotate('Max Length', xy=(max_point['sepal length (cm)'], max_point['sepal width (cm)']),
             xytext=(max_point['sepal length (cm)'] + 0.5, max_point['sepal width (cm)'] + 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.title('Sepal Length vs. Width with Annotation', fontsize=14)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.savefig('annotated_scatter.png')
plt.close()
print("\nAnnotated scatter saved as 'annotated_scatter.png'")

# %% [3. Using Colormaps]
# Apply a colormap to a scatter plot.
np.random.seed(42)
values = np.random.rand(len(df))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=values, cmap='viridis', alpha=0.5)
plt.colorbar(label='Value')
plt.title('Sepal Features with Viridis Colormap', fontsize=14)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.savefig('colormap_scatter.png')
plt.close()
print("Colormap scatter saved as 'colormap_scatter.png'")

# %% [4. Customizing Axes]
# Customize axes with log scale and grid.
np.random.seed(42)
x = np.linspace(1, 100, 100)
y = np.exp(0.05 * x + np.random.normal(0, 0.1, 100))
plt.plot(x, y, color='red', label='Exponential Trend')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.title('Exponential Trend with Log Scale', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y (log scale)')
plt.legend()
plt.savefig('custom_axes.png')
plt.close()
print("Custom axes plot saved as 'custom_axes.png'")

# %% [5. Practical ML Application]
# Customize an ML feature plot.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'target': np.random.choice([0, 1], 100)
})
plt.scatter(ml_data['feature1'], ml_data['feature2'], c=ml_data['target'], cmap='coolwarm', alpha=0.6)
plt.colorbar(label='Class')
plt.annotate('Class 0 Cluster', xy=(10, 5), xytext=(12, 6), arrowprops=dict(facecolor='black', shrink=0.05))
plt.title('ML Features by Class', fontsize=14)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.grid(True, linestyle=':')
plt.savefig('ml_custom_plot.png')
plt.close()
print("ML custom plot saved as 'ml_custom_plot.png'")

# %% [6. Interview Scenario: Customization]
# Discuss advanced customization for ML.
print("\nInterview Scenario: Customization")
print("Q: How would you highlight a key point in a Matplotlib scatter plot?")
print("A: Use plt.annotate to add text and an arrow pointing to the point.")
print("Key: Annotations improve interpretability of ML visualizations.")
print("Example: plt.annotate('Key', xy=(x, y), arrowprops=dict(facecolor='black'))")