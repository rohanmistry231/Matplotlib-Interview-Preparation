import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Integration with Pandas]
# Learn how to use Matplotlib with Pandas for ML data visualization.
# Covers plotting DataFrame columns and visualizing grouped data.

print("Matplotlib version:", plt.matplotlib.__version__)
print("Pandas version:", pd.__version__)

# %% [2. Plotting DataFrame Columns]
# Plot Iris DataFrame columns.
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
df['sepal length (cm)'].plot(kind='hist', bins=20, color='blue', alpha=0.7, title='Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('df_histogram.png')
plt.close()
print("\nDataFrame histogram saved as 'df_histogram.png'")

# %% [3. Plotting Multiple Columns]
# Plot multiple features.
df[['sepal length (cm)', 'sepal width (cm)']].plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)', color='green', alpha=0.5)
plt.title('Sepal Length vs. Width')
plt.savefig('df_scatter.png')
plt.close()
print("DataFrame scatter saved as 'df_scatter.png'")

# %% [4. Visualizing Grouped Data]
# Group by synthetic categories and plot.
np.random.seed(42)
df['category'] = np.random.choice(['A', 'B', 'C'], len(df))
grouped = df.groupby('category')
for name, group in grouped:
    plt.hist(group['sepal length (cm)'], bins=15, alpha=0.5, label=f'Category {name}')
plt.title('Sepal Length by Category')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('grouped_histogram.png')
plt.close()
print("Grouped histogram saved as 'grouped_histogram.png'")

# %% [5. Practical ML Application]
# Visualize ML dataset features with Pandas.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'target': np.random.choice([0, 1], 100)
})
ml_data[ml_data['target'] == 0]['feature1'].plot(kind='hist', bins=15, alpha=0.5, color='blue', label='Class 0')
ml_data[ml_data['target'] == 1]['feature1'].plot(kind='hist', bins=15, alpha=0.5, color='red', label='Class 1')
plt.title('Feature1 Distribution by Class')
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('ml_class_histogram.png')
plt.close()
print("ML class histogram saved as 'ml_class_histogram.png'")

# %% [6. Interview Scenario: Pandas Integration]
# Discuss Pandas integration for ML.
print("\nInterview Scenario: Pandas Integration")
print("Q: How would you visualize a Pandas DataFrame column in Matplotlib?")
print("A: Use df.plot(kind='hist') or df.plot(kind='scatter') for quick visualizations.")
print("Key: Pandas’ plot method simplifies Matplotlib integration.")
print("Example: df['col'].plot(kind='hist', bins=20, color='blue')")