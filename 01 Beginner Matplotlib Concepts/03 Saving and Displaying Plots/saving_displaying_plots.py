import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Saving and Displaying Plots]
# Learn how to save and display Matplotlib plots for ML workflows.
# Covers saving as PNG/PDF/SVG and displaying in Jupyter/scripts.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Saving Plots as PNG]
# Save a scatter plot as PNG.
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
plt.title('Sepal Length vs. Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.savefig('scatter_png.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nScatter plot saved as 'scatter_png.png'")

# %% [3. Saving Plots as PDF]
# Save a histogram as PDF.
plt.hist(df['sepal length (cm)'], bins=20, color='green', alpha=0.7)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('histogram_pdf.pdf', bbox_inches='tight')
plt.close()
print("Histogram saved as 'histogram_pdf.pdf'")

# %% [4. Saving Plots as SVG]
# Save a line plot as SVG.
np.random.seed(42)
days = np.arange(1, 31)
sales = 1000 + np.random.normal(0, 50, 30).cumsum()
plt.plot(days, sales, color='red', linestyle='-')
plt.title('Sales Trend')
plt.xlabel('Day')
plt.ylabel('Sales ($)')
plt.savefig('line_svg.svg', bbox_inches='tight')
plt.close()
print("Line plot saved as 'line_svg.svg'")

# %% [5. Displaying Plots]
# Display a bar plot (commented for non-interactive environments).
categories = ['A', 'B', 'C']
counts = np.random.randint(50, 100, 3)
plt.bar(categories, counts, color='orange')
plt.title('Category Counts')
plt.xlabel('Category')
plt.ylabel('Count')
# plt.show()  # Uncomment to display in interactive environments
plt.savefig('bar_display.png')
plt.close()
print("Bar plot saved as 'bar_display.png' (display commented)")

# %% [6. Practical ML Application]
# Save an ML feature plot for a report.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100)
})
plt.scatter(ml_data['feature1'], ml_data['feature2'], color='purple', alpha=0.6)
plt.title('ML Feature Scatter')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.savefig('ml_feature_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("ML feature scatter saved as 'ml_feature_report.png'")

# %% [7. Interview Scenario: Saving Plots]
# Discuss saving plots for ML.
print("\nInterview Scenario: Saving Plots")
print("Q: How would you save a Matplotlib plot for an ML report?")
print("A: Use plt.savefig with formats like PNG or PDF, adjusting dpi for quality.")
print("Key: Use bbox_inches='tight' to prevent clipping.")
print("Example: plt.savefig('plot.png', dpi=150, bbox_inches='tight')")