import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    make_classification, LogisticRegression, RandomForestClassifier = None, None, None

# %% [1. Introduction to Custom Visualizations]
# Learn how to create custom visualizations for ML insights.
# Covers decision boundaries, feature importance, and custom colormaps.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Plotting Decision Boundaries]
# Visualize decision boundaries for a classifier.
if make_classification:
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)
    model = LogisticRegression().fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.6)
    plt.title('Decision Boundary for Logistic Regression', fontsize=14)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.savefig('decision_boundary.png')
    plt.close()
    print("\nDecision boundary saved as 'decision_boundary.png'")
else:
    print("Scikit-learn unavailable; skipping decision boundary plot.")

# %% [3. Visualizing Feature Importance]
# Plot feature importance for a random forest.
if make_classification and RandomForestClassifier:
    X, y = make_classification(n_features=5, random_state=42)
    model = RandomForestClassifier(random_state=42).fit(X, y)
    importance = model.feature_importances_
    features = [f'Feature {i+1}' for i in range(5)]
    plt.barh(features, importance, color='green', alpha=0.7)
    plt.title('Feature Importance from Random Forest', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance saved as 'feature_importance.png'")
else:
    np.random.seed(42)
    importance = np.random.rand(5)
    features = [f'Feature {i+1}' for i in range(5)]
    plt.barh(features, importance, color='green', alpha=0.7)
    plt.title('Synthetic Feature Importance', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')
    plt.close()
    print("Synthetic feature importance saved as 'feature_importance.png'")

# %% [4. Creating Custom Colormaps]
# Create and use a custom colormap.
from matplotlib.colors import LinearSegmentedColormap
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100),
    'value': np.random.rand(100)
})
colors = ['#FF0000', '#00FF00', '#0000FF']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
plt.scatter(df['feature1'], df['feature2'], c=df['value'], cmap=cmap, alpha=0.6)
plt.colorbar(label='Value')
plt.title('Scatter with Custom Colormap', fontsize=14)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.savefig('custom_colormap.png')
plt.close()
print("Custom colormap scatter saved as 'custom_colormap.png'")

# %% [5. Practical ML Application]
# Combine decision boundary and feature importance.
if make_classification:
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    model = RandomForestClassifier(random_state=42).fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.6)
    plt.title('Random Forest Decision Boundary', fontsize=14)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.savefig('ml_decision_boundary.png')
    plt.close()
    print("ML decision boundary saved as 'ml_decision_boundary.png'")
else:
    print("Scikit-learn unavailable; skipping ML decision boundary plot.")

# %% [6. Interview Scenario: Custom Visualizations]
# Discuss custom visualizations for ML.
print("\nInterview Scenario: Custom Visualizations")
print("Q: How would you visualize a classifier’s decision boundary?")
print("A: Use plt.contourf with a meshgrid and model predictions.")
print("Key: Decision boundaries show how models separate classes.")
print("Example: plt.contourf(xx, yy, Z, cmap='coolwarm')")