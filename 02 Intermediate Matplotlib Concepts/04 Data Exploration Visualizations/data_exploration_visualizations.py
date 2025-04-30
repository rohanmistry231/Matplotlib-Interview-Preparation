import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
    import seaborn as sns
except ImportError:
    load_iris, sns = None, None

# %% [1. Introduction to Data Exploration Visualizations]
# Learn Matplotlib visualizations for exploring ML data.
# Covers box plots, pair plots, and correlation heatmaps.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Box Plots for Outlier Detection]
# Create box plots for Iris features.
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
plt.boxplot([df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'], df['petal width (cm)']],
            labels=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.title('Box Plots of Iris Features', fontsize=14)
plt.ylabel('Value (cm)')
plt.savefig('box_plots.png')
plt.close()
print("\nBox plots saved as 'box_plots.png'")

# %% [3. Pair Plots for Feature Relationships]
# Create pair plots with Seaborn.
if sns and load_iris:
    sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.5})
    plt.suptitle('Pair Plot of Iris Features', y=1.02, fontsize=16)
    plt.savefig('pair_plot.png')
else:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], alpha=0.5, color='blue')
    ax.set_title('Sepal Length vs. Width (Fallback)')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    plt.savefig('pair_plot.png')
plt.close()
print("Pair plot saved as 'pair_plot.png'")

# %% [4. Correlation Heatmaps]
# Create a correlation heatmap.
corr_matrix = df.corr()
if sns:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Iris Features', fontsize=14)
    plt.savefig('correlation_heatmap.png')
else:
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Heatmap (Fallback)')
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix)), corr_matrix.index)
    plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved as 'correlation_heatmap.png'")

# %% [5. Practical ML Application]
# Explore ML dataset with visualizations.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'feature3': np.random.normal(0, 0.5, 100),
    'target': np.random.choice([0, 1], 100)
})
plt.boxplot([ml_data[ml_data['target'] == 0]['feature1'], ml_data[ml_data['target'] == 1]['feature1']],
            labels=['Class 0', 'Class 1'])
plt.title('Feature1 Box Plot by Class', fontsize=14)
plt.ylabel('Feature1')
plt.savefig('ml_box_plot.png')
plt.close()
print("ML box plot saved as 'ml_box_plot.png'")

# %% [6. Interview Scenario: Data Exploration]
# Discuss data exploration visualizations for ML.
print("\nInterview Scenario: Data Exploration")
print("Q: How would you visualize feature relationships in a dataset?")
print("A: Use seaborn.pairplot for scatter and histogram plots or plt.boxplot for distributions.")
print("Key: Pair plots reveal feature correlations; box plots show outliers.")
print("Example: sns.pairplot(df, diag_kind='hist')")