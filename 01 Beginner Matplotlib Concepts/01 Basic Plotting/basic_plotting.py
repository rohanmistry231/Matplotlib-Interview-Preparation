import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Basic Plotting]
# Learn fundamental Matplotlib plotting for ML data visualization.
# Covers line plots, scatter plots, bar plots, and histograms.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Line Plot]
# Create a line plot for a trend (e.g., synthetic time-series).
np.random.seed(42)
days = np.arange(1, 31)
sales = 1000 + np.random.normal(0, 50, 30).cumsum()
plt.plot(days, sales, color='blue', linestyle='-', label='Sales Trend')
plt.xlabel('Day')
plt.ylabel('Sales ($)')
plt.title('Sales Trend Over 30 Days')
plt.legend()
plt.savefig('line_plot.png')
plt.close()
print("\nLine plot saved as 'line_plot.png'")

# %% [3. Scatter Plot]
# Create a scatter plot for feature relationships (e.g., Iris dataset).
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150)
    })
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], color='green', alpha=0.5, label='Data Points')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs. Width')
plt.legend()
plt.savefig('scatter_plot.png')
plt.close()
print("Scatter plot saved as 'scatter_plot.png'")

# %% [4. Bar Plot]
# Create a bar plot for category comparisons.
np.random.seed(42)
categories = ['A', 'B', 'C']
counts = np.random.randint(50, 100, 3)
plt.bar(categories, counts, color='orange', label='Category Counts')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Counts')
plt.legend()
plt.savefig('bar_plot.png')
plt.close()
print("Bar plot saved as 'bar_plot.png'")

# %% [5. Histogram]
# Create a histogram for distribution (e.g., Iris feature).
sepal_lengths = df['sepal length (cm)']
plt.hist(sepal_lengths, bins=20, color='purple', alpha=0.7, label='Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Length')
plt.legend()
plt.savefig('histogram.png')
plt.close()
print("Histogram saved as 'histogram.png'")

# %% [6. Practical ML Application]
# Visualize ML feature distributions.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100)
})
plt.hist(ml_data['feature1'], bins=15, color='blue', alpha=0.5, label='Feature1')
plt.hist(ml_data['feature2'], bins=15, color='red', alpha=0.5, label='Feature2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('ML Feature Distributions')
plt.legend()
plt.savefig('ml_features_histogram.png')
plt.close()
print("ML feature histogram saved as 'ml_features_histogram.png'")

# %% [7. Interview Scenario: Basic Plotting]
# Discuss basic plotting for ML.
print("\nInterview Scenario: Basic Plotting")
print("Q: How would you visualize a feature’s distribution in Matplotlib?")
print("A: Use plt.hist to create a histogram with customizable bins and colors.")
print("Key: Histograms reveal data distributions for ML preprocessing.")
print("Example: plt.hist(df['col'], bins=20, color='blue')")