import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, RocCurveDisplay, confusion_matrix
    import seaborn as sns
except ImportError:
    make_classification, train_test_split, LogisticRegression, roc_curve, RocCurveDisplay, confusion_matrix, sns = None, None, None, None, None, None, None

# %% [1. Introduction to ML Evaluation Plots]
# Learn how to visualize ML model performance with Matplotlib.
# Covers accuracy/loss curves, confusion matrices, and AUC-ROC curves.

print("Matplotlib version:", plt.matplotlib.__version__)

# %% [2. Accuracy and Loss Curves]
# Simulate training metrics.
np.random.seed(42)
epochs = np.arange(1, 11)
train_acc = 0.6 + 0.04 * np.arange(10) + np.random.normal(0, 0.01, 10)
val_acc = 0.55 + 0.035 * np.arange(10) + np.random.normal(0, 0.015, 10)
plt.plot(epochs, train_acc, label='Training Accuracy', color='blue', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red', marker='s')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curves.png')
plt.close()
print("\nAccuracy curves saved as 'accuracy_curves.png'")

# %% [3. Confusion Matrix]
# Create a confusion matrix for a synthetic classifier.
if make_classification:
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
else:
    np.random.seed(42)
    cm = np.array([[50, 10], [15, 25]])
if sns:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
else:
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

# %% [4. AUC-ROC Curve]
# Plot an AUC-ROC curve.
if make_classification and RocCurveDisplay:
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=0.5 * (tpr[-1] + 1))
    display.plot()
    plt.title('AUC-ROC Curve', fontsize=14)
    plt.savefig('roc_curve.png')
else:
    np.random.seed(42)
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Synthetic ROC
    plt.plot(fpr, tpr, label='ROC Curve', color='purple')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Synthetic AUC-ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve.png')
plt.close()
print("AUC-ROC curve saved as 'roc_curve.png'")

# %% [5. Practical ML Application]
# Visualize multiple model metrics.
if make_classification:
    models = {'Logistic Regression': LogisticRegression().fit(X_train, y_train)}
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f'{name} ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Model Comparison: ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('model_roc_comparison.png')
else:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Synthetic ROC', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('Synthetic Model ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('model_roc_comparison.png')
plt.close()
print("Model ROC comparison saved as 'model_roc_comparison.png'")

# %% [6. Interview Scenario: ML Evaluation]
# Discuss ML evaluation plots.
print("\nInterview Scenario: ML Evaluation")
print("Q: How would you plot an AUC-ROC curve in Matplotlib?")
print("A: Use sklearn.metrics.roc_curve and plot fpr vs. tpr with plt.plot.")
print("Key: AUC-ROC evaluates classifier performance for imbalanced data.")
print("Example: fpr, tpr, _ = roc_curve(y_test, y_score); plt.plot(fpr, tpr)")