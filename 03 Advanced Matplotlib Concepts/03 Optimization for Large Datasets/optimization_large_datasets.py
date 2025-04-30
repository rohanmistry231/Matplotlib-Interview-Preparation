import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [1. Introduction to Optimization for Large Datasets]
# Learn how to optimize Matplotlib for large ML datasets.
# Covers downsampling, sparse data plotting, and blitting for animations.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Downsampling Data]
# Downsample a large dataset for faster plotting.
np.random.seed(42)
n_points = 100000
x = np.linspace(0, 10, n_points)
y = np.sin(x) + np.random.normal(0, 0.1, n_points)
downsample_factor = 100
x_down = x[::downsample_factor]
y_down = y[::downsample_factor]
plt.scatter(x_down, y_down, color='blue', alpha=0.5, s=10)
plt.title('Downsampled Scatter Plot', fontsize=14)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('downsampled_scatter.png')
plt.close()
print("\nDownsampled scatter saved as 'downsampled_scatter.png'")

# %% [3. Plotting Sparse Data]
# Plot sparse data efficiently.
np.random.seed(42)
sparse_data = np.zeros(10000)
indices = np.random.choice(10000, 1000, replace=False)
sparse_data[indices] = np.random.normal(0, 1, 1000)
plt.plot(sparse_data, color='red', linestyle='-', alpha=0.7)
plt.title('Sparse Data Plot', fontsize=14)
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('sparse_plot.png')
plt.close()
print("Sparse data plot saved as 'sparse_plot.png'")

# %% [4. Blitting for Animation Performance]
# Use blitting to optimize animation.
np.random.seed(42)
t = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], color='purple', label='Wave')
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Optimized Wave Animation')
ax.legend()
ax.grid(True)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(t, np.sin(t + frame * 0.1))
    return line,

ani = FuncAnimation(fig, update, init_func=init, frames=100, interval=50, blit=True)
ani.save('wave_animation.mp4', writer='ffmpeg')
plt.close()
print("Optimized wave animation saved as 'wave_animation.mp4'")

# %% [5. Practical ML Application]
# Optimize a large ML dataset visualization.
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100000),
    'feature2': np.random.normal(5, 1, 100000),
    'target': np.random.choice([0, 1], 100000)
})
downsample_idx = np.random.choice(len(ml_data), 1000, replace=False)
ml_down = ml_data.iloc[downsample_idx]
plt.scatter(ml_down['feature1'], ml_down['feature2'], c=ml_down['target'], cmap='coolwarm', alpha=0.5, s=10)
plt.title('Downsampled ML Features', fontsize=14)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.colorbar(label='Class')
plt.grid(True)
plt.savefig('ml_downsampled.png')
plt.close()
print("Downsampled ML plot saved as 'ml_downsampled.png'")

# %% [6. Interview Scenario: Optimization]
# Discuss optimization for large datasets.
print("\nInterview Scenario: Optimization")
print("Q: How would you optimize a Matplotlib plot for a large dataset?")
print("A: Downsample data, use sparse plotting, or enable blitting for animations.")
print("Key: Downsampling reduces rendering time while preserving trends.")
print("Example: x_down = x[::100]; plt.scatter(x_down, y_down)")