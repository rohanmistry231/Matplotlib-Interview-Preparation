# 🏋️ Intermediate Matplotlib Concepts (`matplotlib`)

## 📖 Introduction
Matplotlib is a powerful tool for visualizing AI and machine learning (ML) model performance and data insights. This section deepens your Matplotlib skills with **Subplots and Layouts**, **ML Evaluation Plots**, **Advanced Customization**, and **Data Exploration Visualizations**, building on beginner concepts (e.g., basic plotting, customization). With practical examples and interview insights, it prepares you to create advanced visualizations for ML evaluation and data exploration, complementing Pandas and NumPy skills.

## 🎯 Learning Objectives
- Create and manage subplot grids for multi-faceted visualizations.
- Visualize ML model performance with accuracy curves, confusion matrices, and AUC-ROC curves.
- Apply advanced customization like annotations, colormaps, and axis scaling.
- Explore datasets with box plots, pair plots, and correlation heatmaps.

## 🔑 Key Concepts
- **Subplots and Layouts**:
  - Creating multiple plots (`plt.subplots`) in a grid.
  - Adjusting subplot spacing (`plt.tight_layout`).
  - Sharing axes for consistent scales.
- **ML Evaluation Plots**:
  - Accuracy and loss curves for model training (`plt.plot`).
  - Confusion matrices using `seaborn.heatmap` or `plt.imshow`.
  - AUC-ROC curves with scikit-learn (`RocCurveDisplay`).
- **Advanced Customization**:
  - Annotating plots (`plt.annotate`) for key points.
  - Using colormaps (`cmap`) for heatmaps and scatters.
  - Customizing axes (`plt.yscale`, `plt.grid`).
- **Data Exploration Visualizations**:
  - Box plots (`plt.boxplot`) for outlier detection.
  - Pair plots (`seaborn.pairplot`) for feature relationships.
  - Correlation heatmaps (`plt.imshow`, `seaborn.heatmap`).

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`subplots_layouts.py`**:
   - Creates a 2x2 subplot grid of Iris feature histograms (`plt.subplots`).
   - Adjusts spacing with `plt.tight_layout` for mixed plots (scatter, line).
   - Uses shared axes for consistent scales in histograms.
   - Visualizes ML feature distributions by class in subplots.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   fig, axes = plt.subplots(2, 2, figsize=(10, 8))
   axes[0, 0].hist(df['col'], bins=20)
   plt.tight_layout()
   ```

2. **`ml_evaluation_plots.py`**:
   - Plots training/validation accuracy curves (`plt.plot`).
   - Creates a confusion matrix for a classifier (`seaborn.heatmap` or `plt.imshow`).
   - Generates an AUC-ROC curve using scikit-learn (`RocCurveDisplay`).
   - Compares ROC curves for multiple models.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   from sklearn.metrics import roc_curve
   fpr, tpr, _ = roc_curve(y_test, y_score)
   plt.plot(fpr, tpr, label='ROC')
   ```

3. **`advanced_customization.py`**:
   - Annotates a scatter plot to highlight key points (`plt.annotate`).
   - Applies a colormap to a scatter plot (`cmap='viridis'`).
   - Customizes axes with log scale and grid (`plt.yscale`, `plt.grid`).
   - Visualizes ML features by class with customizations.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(df['col1'], df['col2'], c=values, cmap='viridis')
   plt.annotate('Key', xy=(x, y), arrowprops=dict(facecolor='black'))
   ```

4. **`data_exploration_visualizations.py`**:
   - Creates box plots for Iris features (`plt.boxplot`).
   - Generates pair plots with Seaborn (`seaborn.pairplot`) or fallback scatter.
   - Plots a correlation heatmap (`seaborn.heatmap` or `plt.imshow`).
   - Visualizes ML feature distributions by class with box plots.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   plt.boxplot([df['col1'], df['col2']], labels=['Col1', 'Col2'])
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   ```

## 🛠️ Practical Tasks
1. **Subplots and Layouts**:
   - Create a 2x2 subplot grid of feature histograms.
   - Plot scatter and line plots with adjusted spacing (`plt.tight_layout`).
   - Use shared axes to compare two feature distributions.
2. **ML Evaluation Plots**:
   - Plot training and validation accuracy curves for a model.
   - Create a confusion matrix heatmap for a classifier.
   - Generate an AUC-ROC curve using scikit-learn.
3. **Advanced Customization**:
   - Annotate a scatter plot to highlight the maximum value.
   - Apply a colormap to a scatter plot of ML features.
   - Use a log scale for a plot with exponential data.
4. **Data Exploration Visualizations**:
   - Create box plots for all features in a dataset.
   - Generate a pair plot to explore feature relationships.
   - Plot a correlation heatmap for a DataFrame.

## 💡 Interview Tips
- **Common Questions**:
  - How do you create a grid of subplots in Matplotlib?
  - How would you plot an AUC-ROC curve for a classifier?
  - What’s the benefit of using a colormap in a scatter plot?
  - How do you visualize feature correlations in a dataset?
- **Tips**:
  - Explain `plt.subplots` for organizing multiple plots.
  - Highlight AUC-ROC for evaluating imbalanced classifiers.
  - Be ready to code a confusion matrix or pair plot.
- **Coding Tasks**:
  - Create a subplot grid of feature distributions.
  - Plot a ROC curve for a binary classifier.
  - Generate a correlation heatmap for ML features.

## 📚 Resources
- [Matplotlib Subplots](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
- [Scikit-learn Visualization](https://scikit-learn.org/stable/visualizations.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
- [Kaggle: Data Visualization](https://www.kaggle.com/learn/data-visualization)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)