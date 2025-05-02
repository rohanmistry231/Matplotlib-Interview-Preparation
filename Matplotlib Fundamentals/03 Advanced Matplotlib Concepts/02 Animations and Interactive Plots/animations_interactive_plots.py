import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
try:
    import mplcursors
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    mplcursors, tk, FigureCanvasTkAgg = None, None, None

# %% [1. Introduction to Animations and Interactive Plots]
# Learn how to create dynamic and interactive visualizations for ML.
# Covers animations, interactive plots, and GUI embedding.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Creating Animations with FuncAnimation]
# Animate a loss curve.
np.random.seed(42)
epochs = np.arange(1, 21)
loss = 1 / (1 + 0.5 * epochs) + np.random.normal(0, 0.02, 20)
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], label='Training Loss', color='blue')
ax.set_xlim(1, 20)
ax.set_ylim(0, 1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Animated Loss Curve')
ax.legend()
ax.grid(True)

def update(frame):
    line.set_data(epochs[:frame], loss[:frame])
    return line,

ani = FuncAnimation(fig, update, frames=range(1, 21), interval=200, blit=True)
ani.save('loss_animation.mp4', writer='ffmpeg')
plt.close()
print("\nLoss animation saved as 'loss_animation.mp4'")

# %% [3. Interactive Plots with mplcursors]
# Create an interactive scatter plot.
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 100),
    'feature2': np.random.normal(5, 1, 100)
})
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['feature1'], df['feature2'], color='purple', alpha=0.6)
ax.set_title('Interactive Feature Scatter')
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
if mplcursors:
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"({df['feature1'].iloc[sel.index]:.2f}, {df['feature2'].iloc[sel.index]:.2f})"))
plt.savefig('interactive_scatter.png')
plt.close()
print("Interactive scatter saved as 'interactive_scatter.png'")

# %% [4. Embedding Plots in Tkinter GUI]
# Embed a plot in a Tkinter window (commented for non-GUI environments).
if tk and FigureCanvasTkAgg:
    root = tk.Tk()
    root.title("Matplotlib in Tkinter")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, loss, color='red', label='Loss')
    ax.set_title('Loss Curve in GUI')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    # root.mainloop()  # Uncomment to run GUI
    plt.savefig('tkinter_plot.png')
    plt.close()
    print("Tkinter plot saved as 'tkinter_plot.png'")
else:
    print("Tkinter or FigureCanvasTkAgg unavailable; skipping GUI plot.")

# %% [5. Practical ML Application]
# Animate ML training process.
np.random.seed(42)
epochs = np.arange(1, 21)
train_acc = 0.6 + 0.04 * np.arange(20) + np.random.normal(0, 0.01, 20)
val_acc = 0.55 + 0.035 * np.arange(20) + np.random.normal(0, 0.015, 20)
fig, ax = plt.subplots(figsize=(8, 6))
line1, = ax.plot([], [], label='Training Accuracy', color='blue')
line2, = ax.plot([], [], label='Validation Accuracy', color='red')
ax.set_xlim(1, 20)
ax.set_ylim(0, 1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Animated ML Training Accuracy')
ax.legend()
ax.grid(True)

def update(frame):
    line1.set_data(epochs[:frame], train_acc[:frame])
    line2.set_data(epochs[:frame], val_acc[:frame])
    return line1, line2

ani = FuncAnimation(fig, update, frames=range(1, 21), interval=200, blit=True)
ani.save('ml_training_animation.mp4', writer='ffmpeg')
plt.close()
print("ML training animation saved as 'ml_training_animation.mp4'")

# %% [6. Interview Scenario: Animations]
# Discuss animations for ML.
print("\nInterview Scenario: Animations")
print("Q: How would you animate a training process in Matplotlib?")
print("A: Use FuncAnimation to update plot data over frames.")
print("Key: Animations visualize dynamic ML processes like training.")
print("Example: ani = FuncAnimation(fig, update, frames=range(20), interval=200)")