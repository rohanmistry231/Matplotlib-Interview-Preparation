import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
try:
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
except ImportError:
    load_iris, PCA = None, None

# %% [1. Introduction to 3D Visualizations]
# Learn how to create 3D visualizations for ML data with Matplotlib.
# Covers 3D scatter plots, surface plots, and axis customization.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. 3D Scatter Plot]
# Create a 3D scatter plot for Iris features.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    labels = iris.target
else:
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
        'petal length (cm)': np.random.normal(3.7, 1.8, 150)
    })
    labels = np.random.choice([0, 1, 2], 150)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'],
                    c=labels, cmap='viridis', alpha=0.6)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_title('3D Scatter of Iris Features')
plt.colorbar(scatter, label='Class')
plt.savefig('3d_scatter.png')
plt.close()
print("\n3D scatter plot saved as '3d_scatter.png'")

# %% [3. 3D Surface Plot]
# Create a 3D surface plot for synthetic data.
np.random.seed(42)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')
plt.colorbar(surf, label='Value')
plt.savefig('3d_surface.png')
plt.close()
print("3D surface plot saved as '3d_surface.png'")

# %% [4. Visualizing High-Dimensional Data with PCA]
# Use PCA to project data into 3D.
if load_iris and PCA:
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(df)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='plasma', alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA Projection of Iris Features')
    plt.colorbar(scatter, label='Class')
    plt.savefig('pca_3d.png')
    plt.close()
    print("PCA 3D plot saved as 'pca_3d.png'")
else:
    print("PCA or Iris dataset unavailable; skipping PCA plot.")

# %% [5. Customizing 3D Axes and Viewpoints]
# Customize 3D plot with viewpoint.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'],
           c=labels, cmap='viridis', alpha=0.6)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_title('Custom 3D View')
ax.view_init(elev=30, azim=45)
plt.savefig('custom_3d_view.png')
plt.close()
print("Custom 3D view saved as 'custom_3d_view.png'")

# %% [6. Practical ML Application]
# Visualize ML features in 3D.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'feature3': np.random.normal(0, 0.5, 100),
    'target': np.random.choice([0, 1], 100)
})
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for target in [0, 1]:
    subset = ml_data[ml_data['target'] == target]
    ax.scatter(subset['feature1'], subset['feature2'], subset['feature3'],
               label=f'Class {target}', alpha=0.6)
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Feature3')
ax.set_title('3D ML Feature Visualization')
ax.legend()
plt.savefig('ml_3d_features.png')
plt.close()
print("ML 3D features saved as 'ml_3d_features.png'")

# %% [7. Interview Scenario: 3D Visualizations]
# Discuss 3D visualizations for ML.
print("\nInterview Scenario: 3D Visualizations")
print("Q: How would you visualize high-dimensional ML data in Matplotlib?")
print("A: Use PCA to reduce dimensions and plot a 3D scatter with Axes3D.")
print("Key: 3D plots reveal data structure; PCA simplifies visualization.")
print("Example: ax = fig.add_subplot(111, projection='3d'); ax.scatter(x, y, z)")