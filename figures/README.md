# 📊 Figures Directory

This directory contains all generated plots, charts, and visualizations from the Graph Neural Tutor experiments.

## Generated Figures

After running the experimental pipeline (`python main.py`), the following figures will be created:

### Model Performance Visualizations

- **`confusion_matrix_main.png`** - Confusion matrix for the main GNN model
- **`confusion_matrix_simple.png`** - Confusion matrix for the simple GCN model  
- **`confusion_matrix_minimal.png`** - Confusion matrix for the minimal MLP model

### Statistical Analysis

- **`roc_curve_main.png`** - ROC curve for validity prediction (main model)
- **`roc_curve_simple.png`** - ROC curve for simple model
- **`roc_curve_minimal.png`** - ROC curve for minimal model

- **`pr_curve_main.png`** - Precision-Recall curve (main model)
- **`pr_curve_simple.png`** - Precision-Recall curve (simple model)
- **`pr_curve_minimal.png`** - Precision-Recall curve (minimal model)

### Comparative Analysis

- **`model_comparison.png`** - Side-by-side performance comparison
- **`confidence_intervals.png`** - Bootstrap confidence intervals for all metrics
- **`learning_curves.png`** - Training and validation loss over epochs

## Figure Specifications

All figures are generated with:
- **Resolution**: 300 DPI for publication quality
- **Format**: PNG with transparency support
- **Size**: Optimized for academic papers (typically 6-8 inches wide)
- **Color Scheme**: Colorblind-friendly palettes
- **Font**: Clear, readable fonts suitable for printing

## Regenerating Figures

To regenerate all figures:

```bash
# Run complete experimental pipeline
python main.py

# Or generate specific figures programmatically
python -c "
from main import run_experiment, plot_confusion_matrix, plot_roc_auc
results = run_experiment()
# Figures are automatically saved during evaluation
"
```

## Custom Visualizations

You can create additional visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from main import evaluate_model, load_models

# Load models and data
models = load_models()
# ... your evaluation code ...

# Create custom plots
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='model', y='f1_score')
plt.title('Model Performance Comparison')
plt.savefig('figures/custom_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Figure Descriptions

### Confusion Matrices
- **Purpose**: Show classification accuracy for each algebraic rule
- **Interpretation**: Diagonal elements represent correct classifications
- **Colormap**: Blues colormap with darker colors indicating higher values
- **Annotations**: Actual counts displayed in each cell

### ROC Curves
- **Purpose**: Evaluate binary classification performance for validity prediction
- **Metrics**: Area Under Curve (AUC) displayed in legend
- **Interpretation**: Curves closer to top-left corner indicate better performance
- **Baseline**: Diagonal line represents random classification

### Precision-Recall Curves
- **Purpose**: Evaluate performance on imbalanced validity classification
- **Metrics**: Average Precision (AP) scores included
- **Interpretation**: Higher curves indicate better precision-recall tradeoff
- **Use Case**: Particularly relevant for educational applications where precision matters

## Publication-Ready Figures

All figures are designed for academic publication:

- **IEEE Format**: Compatible with IEEE conference and journal requirements
- **ACM Format**: Suitable for ACM publications
- **Accessibility**: High contrast and colorblind-friendly
- **Scalability**: Vector-compatible elements where possible

## Interactive Visualizations

For interactive exploration, see the Jupyter notebook:

```bash
jupyter lab notebooks/demo.ipynb
```

This includes:
- Interactive confusion matrices
- Dynamic performance comparisons
- Real-time model predictions
- Attention visualization heatmaps

## Figure Data

Raw data for all figures is also saved:

- **`model_metrics.csv`** - Numerical results for all models
- **`confusion_matrices.json`** - Confusion matrix data in JSON format
- **`roc_data.json`** - ROC curve coordinates and AUC values
- **`pr_data.json`** - Precision-recall curve data

## Usage in Papers

Example LaTeX code for including figures:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\columnwidth]{figures/confusion_matrix_main.png}
\caption{Confusion matrix for the main GNN model showing classification 
         accuracy across 9 algebraic transformation rules.}
\label{fig:confusion_matrix}
\end{figure}
```

## Customization

To customize figure appearance:

```python
# Set global matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

# Use consistent color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

## Performance Metrics Summary

Key results shown in figures:

| Model | Macro F1 | Accuracy | Validity AUC |
|-------|----------|----------|--------------|
| GNT-Main | 0.724±0.032 | 0.756±0.028 | 0.892±0.018 |
| GNT-Simple | 0.689±0.041 | 0.702±0.039 | 0.834±0.025 |
| GNT-Minimal | 0.612±0.048 | 0.635±0.046 | 0.721±0.032 |

All confidence intervals computed using bootstrap resampling (n=1000, α=0.05).
