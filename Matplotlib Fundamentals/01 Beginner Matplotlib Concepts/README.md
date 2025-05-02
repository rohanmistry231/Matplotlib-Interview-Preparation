# 🌱 Beginner Matplotlib Concepts (`matplotlib`)

## 📖 Introduction
Matplotlib is the foundational Python library for data visualization, critical for exploring datasets and communicating insights in AI and machine learning (ML). This section introduces the fundamentals of Matplotlib, focusing on **Basic Plotting**, **Plot Customization**, **Saving and Displaying Plots**, and **Integration with Pandas**. With practical examples and interview insights, it prepares beginners to visualize ML data effectively, complementing Pandas and NumPy skills.

## 🎯 Learning Objectives
- Create basic plots (line, scatter, bar, histogram) for data exploration.
- Customize plots with titles, labels, colors, and figure settings.
- Save plots in various formats (PNG, PDF, SVG) and display them in scripts.
- Integrate Matplotlib with Pandas for DataFrame visualizations.

## 🔑 Key Concepts
- **Basic Plotting**:
  - Line plots (`plt.plot`) for trends.
  - Scatter plots (`plt.scatter`) for relationships.
  - Bar plots (`plt.bar`) for comparisons.
  - Histograms (`plt.hist`) for distributions.
- **Plot Customization**:
  - Setting titles, labels, legends (`plt.title`, `plt.xlabel`, `plt.legend`).
  - Customizing colors, markers, line styles.
  - Adjusting figure size and resolution (`plt.figure(figsize)`).
- **Saving and Displaying Plots**:
  - Saving plots (`plt.savefig`) as PNG, PDF, or SVG.
  - Displaying plots (`plt.show`) in Jupyter or scripts.
- **Integration with Pandas**:
  - Plotting DataFrame columns (`df.plot`).
  - Visualizing grouped data (`df.groupby().plot`).

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`basic_plotting.py`**:
   - Creates a line plot of synthetic sales trends (`plt.plot`).
   - Plots a scatter of Iris sepal features (`plt.scatter`).
   - Generates a bar plot of category counts (`plt.bar`) and histogram of sepal lengths (`plt.hist`).
   - Visualizes ML feature distributions.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   plt.plot(days, sales, color='blue', label='Sales')
   plt.hist(df['sepal length (cm)'], bins=20, color='purple')
   plt.savefig('plot.png')
   ```

2. **`plot_customization.py`**:
   - Customizes a scatter plot with titles, labels, and legends (`plt.title`, `plt.legend`).
   - Adjusts colors, markers, and line styles in a line plot.
   - Sets figure size and resolution (`plt.figure(figsize)`).
   - Visualizes ML feature comparisons.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(df['col1'], df['col2'], color='blue', label='Data')
   plt.title('Plot', fontsize=14)
   plt.figure(figsize=(10, 6))
   ```

3. **`saving_displaying_plots.py`**:
   - Saves a scatter plot as PNG (`plt.savefig`, `dpi=100`).
   - Saves a histogram as PDF and a line plot as SVG.
   - Demonstrates displaying plots (`plt.show`, commented).
   - Saves an ML feature plot for a report.

   Example code:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(df['col1'], df['col2'])
   plt.savefig('plot.png', dpi=150, bbox_inches='tight')
   ```

4. **`integration_pandas.py`**:
   - Plots DataFrame columns with `df.plot(kind='hist')` and `df.plot(kind='scatter')`.
   - Visualizes grouped data by categories (`df.groupby`).
   - Creates histograms of ML features by class.
   - Saves all plots as PNG.

   Example code:
   ```python
   import pandas as pd
   df['col'].plot(kind='hist', bins=20, color='blue')
   df.groupby('category')['col'].plot(kind='hist', alpha=0.5)
   ```

## 🛠️ Practical Tasks
1. **Basic Plotting**:
   - Create a line plot of a synthetic time-series (e.g., sales over days).
   - Plot a histogram of an Iris feature (e.g., sepal length).
   - Generate a scatter plot of two ML features.
2. **Plot Customization**:
   - Customize a scatter plot with a title, labels, and legend.
   - Adjust colors and markers in a line plot.
   - Create a large figure (10x6 inches) for a histogram.
3. **Saving and Displaying**:
   - Save a bar plot as PNG with high resolution (`dpi=150`).
   - Save a line plot as PDF for a report.
   - Display a plot in a Jupyter notebook (use `plt.show`).
4. **Pandas Integration**:
   - Plot a DataFrame column’s distribution with `df.plot(kind='hist')`.
   - Create a scatter plot of two DataFrame columns.
   - Visualize feature distributions by ML class using `groupby`.

## 💡 Interview Tips
- **Common Questions**:
  - How do you create a histogram in Matplotlib?
  - What’s the difference between `plt.plot` and `plt.scatter`?
  - How do you save a Matplotlib plot for a report?
  - How do you visualize a Pandas DataFrame column?
- **Tips**:
  - Explain `plt.hist` for distributions and `plt.scatter` for relationships.
  - Highlight `plt.savefig` with `bbox_inches='tight'` to avoid clipping.
  - Be ready to code a simple plot (e.g., `plt.plot(x, y, color='blue')`).
- **Coding Tasks**:
  - Plot a feature’s distribution from a DataFrame.
  - Customize a scatter plot with labels and colors.
  - Save a histogram as PNG for a presentation.

## 📚 Resources
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Matplotlib User Guide](https://matplotlib.org/stable/users/index.html)
- [Matplotlib Plotting Basics](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [Kaggle: Data Visualization](https://www.kaggle.com/learn/data-visualization)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)